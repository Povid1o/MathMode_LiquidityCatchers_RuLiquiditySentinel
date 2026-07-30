from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import os
import ssl
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.downloaders.common import CHUNK_SIZE, USER_AGENT

# Настройки TLS читаются из окружения, поэтому подхватываем .env — иначе кнопка
# «Полное обновление» в дашборде не увидит переменные, заданные в файле.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


BASE_URL = "https://roskazna.gov.ru"
SOURCE_URL = (
    "https://roskazna.gov.ru/finansovye-operacii/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/"
    "razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah"
)
RAW_DIR = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_deposits"
PAGES_DIR = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_pages"
LINKS_FILE = PROJECT_ROOT / "data/raw/treasury_funds/roskazna_deposit_links.txt"
DEFAULT_START_YEAR = 2021
DEFAULT_MAX_PAGES_PER_YEAR = 80
ARCHIVE_MARKER = 'id="start-files-list"'

# Сколько страниц подряд может не скачаться, прежде чем бросим текущий год.
# Одиночный сбой страницы не должен обрывать весь год (раньше был break).
MAX_CONSECUTIVE_PAGE_FAILURES = 3

# roskazna.gov.ru отдаёт цепочку, подписанную корневым CA Минцифры
# («Russian Trusted Root CA»). Его нет ни в системном хранилище, ни в certifi,
# и не будет — поэтому нужен явный PEM-бандл с этим корнем. Путь берётся из
# переменной среды ROSKAZNA_CA_BUNDLE, иначе из certs/ в корне проекта.
# Полное отключение проверки (ssl._create_unverified_context) убрано намеренно:
# источник кормит казначейскую модель, и молчаливое доверие любому сертификату
# здесь опаснее, чем упавший шаг обновления.
CA_BUNDLE_ENV_VAR = "ROSKAZNA_CA_BUNDLE"
DEFAULT_CA_BUNDLE = PROJECT_ROOT / "certs" / "russian_trusted_root_ca.pem"

# Аварийный режим на случай, когда штатная проверка невозможна (у Росказны истёк
# сертификат, а корня Минцифры в хранилище нет), но данные нужны сейчас.
# В переменную кладётся SHA-256 отпечаток ожидаемого сертификата сервера. Цепочка
# доверия при этом не проверяется, но подмена на ЛЮБОЙ другой сертификат отбивается —
# в отличие от полного отключения проверки, где принимается что угодно.
# Осознанный компромисс: пиннинг не защищает от использования скомпрометированного
# и уже отозванного сертификата, срок действия тоже не проверяется.
# Отпечаток снимается так (сверьте значение по независимому каналу):
#   echo | openssl s_client -connect roskazna.gov.ru:443 -servername roskazna.gov.ru \
#     | openssl x509 -outform DER | openssl dgst -sha256
PINNED_CERT_ENV_VAR = "ROSKAZNA_PINNED_CERT_SHA256"

# Сколько раз идти по редиректу внутри того же хоста при пиннинге
MAX_PINNED_REDIRECTS = 3

_pin_notice_shown = False


class RoskaznaTlsError(RuntimeError):
    """TLS-сертификат источника не проверяется: нет доверенного корня или он истёк.

    Отдельный тип нужен, чтобы отличать «весь хост недоступен» (нет смысла
    перебирать 80 страниц) от разового сбоя на одной странице.
    """


def resolve_ca_bundle() -> Path | None:
    """Возвращает путь к PEM-бандлу с корневым CA Минцифры, если он есть"""
    configured = os.environ.get(CA_BUNDLE_ENV_VAR, "").strip()
    candidate = Path(configured) if configured else DEFAULT_CA_BUNDLE
    return candidate if candidate.is_file() else None


def resolve_pinned_fingerprint() -> str | None:
    """Возвращает нормализованный SHA-256 отпечаток из переменной среды, если задан"""
    raw = os.environ.get(PINNED_CERT_ENV_VAR, "").strip()
    if not raw:
        return None

    normalized = raw.replace(":", "").replace(" ", "").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise RoskaznaTlsError(
            f"{PINNED_CERT_ENV_VAR} должен содержать SHA-256 отпечаток из 64 hex-символов, "
            f"получено: {raw!r}"
        )
    return normalized


def _announce_pinned_mode(fingerprint: str) -> None:
    """Один раз за процесс сообщает, что работаем в режиме пиннинга"""
    global _pin_notice_shown
    if _pin_notice_shown:
        return
    _pin_notice_shown = True
    print(
        "Росказна: цепочка доверия НЕ проверяется, сертификат сверяется по отпечатку "
        f"{fingerprint[:16]}… (режим {PINNED_CERT_ENV_VAR}). "
        "Срок действия и отзыв не проверяются — снимите переменную, как только "
        "источник перевыпустит сертификат."
    )


def _download_with_pinned_certificate(
    url: str,
    output_path: Path,
    fingerprint: str,
    *,
    redirects_left: int = MAX_PINNED_REDIRECTS,
) -> None:
    """Качает файл, сверяя сертификат сервера с ожидаемым отпечатком.

    Отпечаток проверяется на том же соединении, по которому идёт загрузка,
    поэтому подменить сертификат между проверкой и скачиванием нельзя.
    """
    _announce_pinned_mode(fingerprint)

    parsed = urlparse(url)
    if parsed.hostname is None:
        raise ValueError(f"Не удалось определить хост из ссылки: {url}")

    context = ssl._create_unverified_context()  # проверка заменена сверкой отпечатка
    connection = http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=60, context=context
    )

    try:
        connection.connect()
        peer_der = connection.sock.getpeercert(binary_form=True)
        actual = hashlib.sha256(peer_der).hexdigest()
        if not hmac.compare_digest(actual, fingerprint):
            raise RoskaznaTlsError(
                "Отпечаток сертификата Росказны не совпал с закреплённым. "
                f"Ожидался {fingerprint}, получен {actual}. "
                "Либо источник перевыпустил сертификат — тогда сверьте новый отпечаток "
                f"и обновите {PINNED_CERT_ENV_VAR}, либо соединение перехвачено. "
                "Загрузка остановлена."
            )

        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        connection.request("GET", target, headers={"User-Agent": USER_AGENT})
        response = connection.getresponse()

        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            if not location or redirects_left <= 0:
                raise RuntimeError(f"Редирект без цели или их слишком много: {url}")
            next_url = urljoin(url, location)
            if urlparse(next_url).hostname != parsed.hostname:
                raise RoskaznaTlsError(
                    f"Редирект уводит с {parsed.hostname} на "
                    f"{urlparse(next_url).hostname}: при пиннинге это не разрешено"
                )
            connection.close()
            _download_with_pinned_certificate(
                next_url, output_path, fingerprint, redirects_left=redirects_left - 1
            )
            return

        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status} по ссылке {url}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with temporary_path.open("wb") as file:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    file.write(chunk)
            temporary_path.replace(output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    finally:
        connection.close()


def _build_ssl_context() -> ssl.SSLContext:
    """Строит SSL-контекст с проверкой сертификата (плюс корень Минцифры)"""
    bundle = resolve_ca_bundle()
    if bundle is None:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=str(bundle))


def _tls_hint(error: BaseException) -> str:
    """Переводит ошибку проверки сертификата в понятное действие"""
    text = str(error)
    bundle = resolve_ca_bundle()

    if "certificate has expired" in text or "CERTIFICATE_VERIFY_FAILED] certificate has expired" in text:
        return (
            "сертификат roskazna.gov.ru истёк. Это сторона источника — дождитесь "
            "перевыпуска. Обходить проверку нельзя: истёкший сертификат неотличим "
            "от подменённого."
        )

    if bundle is None:
        return (
            "нет доверенного корневого сертификата. Цепочка roskazna.gov.ru подписана "
            "корнем «Russian Trusted Root CA» (Минцифры), которого нет в системном "
            f"хранилище и в certifi. Положите PEM с этим корнем в {DEFAULT_CA_BUNDLE} "
            f"или укажите путь в {CA_BUNDLE_ENV_VAR}, сверив отпечаток из официального "
            "источника перед установкой."
        )

    return (
        f"бандл {bundle} не проверяет цепочку источника. Убедитесь, что в нём именно "
        "корень «Russian Trusted Root CA» и что он не истёк."
    )


class _XmlLinkParser(HTMLParser):
    """Достает XML-ссылки Росказны из сохраненной HTML-страницы"""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Обрабатывает открывающие HTML-теги"""
        if tag != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")
        if href is None or not href.lower().endswith(".xml"):
            return

        self.links.append(urljoin(BASE_URL, href))


def _xml_output_path(url: str, raw_dir: Path) -> Path:
    """Строит путь сохранения XML-файла по ссылке"""
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name
    if not filename:
        raise ValueError(f"Не удалось определить имя XML-файла из ссылки: {url}")
    return raw_dir / filename


def _page_url(year: int, page: int) -> str:
    """Собирает ссылку на страницу архива Росказны"""
    if page == 1:
        return f"{SOURCE_URL}?filter_year={year}"
    return f"{SOURCE_URL}?filter_year={year}&page={page}"


def _page_output_path(year: int, page: int, pages_dir: Path) -> Path:
    """Строит путь сохранения HTML-страницы Росказны"""
    return pages_dir / f"{year}_page_{page:02d}.html"


def _download_roskazna_file(url: str, output_path: Path) -> None:
    """Скачивает файл Росказны с проверкой TLS-сертификата.

    Если задан ROSKAZNA_PINNED_CERT_SHA256, вместо проверки цепочки сертификат
    сверяется с закреплённым отпечатком (аварийный режим, см. certs/README.md).
    Иначе цепочка проверяется штатно, а ошибку поднимаем как RoskaznaTlsError
    с конкретным действием: иначе она тонет в общем «не удалось скачать».
    """
    pinned = resolve_pinned_fingerprint()
    if pinned is not None:
        _download_with_pinned_certificate(url, output_path, pinned)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(request, timeout=60, context=_build_ssl_context()) as response:
            with temporary_path.open("wb") as file:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    file.write(chunk)
    except (ssl.SSLCertVerificationError, ssl.SSLError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RoskaznaTlsError(f"{_tls_hint(error)} Исходная ошибка: {error}") from error
    except URLError as error:
        temporary_path.unlink(missing_ok=True)
        if isinstance(error.reason, ssl.SSLError):
            raise RoskaznaTlsError(
                f"{_tls_hint(error.reason)} Исходная ошибка: {error.reason}"
            ) from error
        raise

    temporary_path.replace(output_path)


def _read_links_from_text(text: str) -> list[str]:
    """Читает XML-ссылки из HTML-текста"""
    parser = _XmlLinkParser()
    parser.feed(text)
    return parser.links


def _read_links_from_html(path: Path) -> list[str]:
    """Читает XML-ссылки из HTML-файла Росказны"""
    return _read_links_from_text(path.read_text(encoding="utf-8"))


def _read_archive_links_from_html(path: Path) -> list[str]:
    """Читает XML-ссылки из архивной таблицы Росказны"""
    text = path.read_text(encoding="utf-8")
    marker_position = text.find(ARCHIVE_MARKER)
    if marker_position == -1:
        return _read_links_from_text(text)
    return _read_links_from_text(text[marker_position:])


def _read_current_links_from_html(path: Path) -> list[str]:
    """Читает XML-ссылки из верхнего текущего блока Росказны"""
    text = path.read_text(encoding="utf-8")
    marker_position = text.find(ARCHIVE_MARKER)
    if marker_position == -1:
        return []
    return _read_links_from_text(text[:marker_position])


def _is_last_archive_page(path: Path) -> bool:
    """Проверяет, что страница является последней страницей архива"""
    text = path.read_text(encoding="utf-8")
    marker_position = text.find(ARCHIVE_MARKER)
    if marker_position == -1:
        return False

    archive_text = text[marker_position:]
    return (
        'class="page-item disabled" aria-disabled="true" aria-label="Вперёд &raquo;"'
        in archive_text
    )


def _read_links_from_txt(path: Path) -> list[str]:
    """Читает XML-ссылки из текстового файла"""
    if not path.exists():
        return []

    links: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        link = line.strip()
        if link and not link.startswith("#") and link.lower().endswith(".xml"):
            links.append(link)
    return links


def download_roskazna_html_pages(
    years: list[int],
    pages_dir: Path = PAGES_DIR,
    max_pages_per_year: int = DEFAULT_MAX_PAGES_PER_YEAR,
    force: bool = False,
) -> tuple[list[Path], int]:
    """Скачивает HTML-страницы архива Росказны по годам и страницам.

    Возвращает (страницы с XML-ссылками, сколько страниц реально скачано в этом
    прогоне). Второе число нужно вызывающему, чтобы отличить «источник ответил»
    от «взяли всё из кеша» — на этой разнице раньше и терялся сбой сети.

    RoskaznaTlsError не перехватываем: если сертификат не проверяется, он не
    проверится и на остальных 79 страницах.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages_with_links: list[Path] = []
    fetched_count = 0

    for year in years:
        previous_link_sets: set[tuple[str, ...]] = set()
        consecutive_failures = 0

        for page in range(1, max_pages_per_year + 1):
            output_path = _page_output_path(year, page, pages_dir)

            if output_path.exists() and not force:
                print(f"HTML-страница Росказны уже есть: {output_path.name}")
            else:
                try:
                    _download_roskazna_file(_page_url(year, page), output_path)
                    print(f"Скачана HTML-страница Росказны: {output_path.name}")
                    fetched_count += 1
                    consecutive_failures = 0
                except RoskaznaTlsError:
                    raise
                except Exception as error:
                    consecutive_failures += 1
                    print(f"Не удалось скачать HTML-страницу Росказны: {year}, page={page}")
                    print(f"Причина: {error}")
                    if consecutive_failures >= MAX_CONSECUTIVE_PAGE_FAILURES:
                        print(
                            f"Останавливаем {year}: {consecutive_failures} страниц подряд "
                            "не скачались"
                        )
                        break
                    # разовый сбой страницы не должен обрывать год
                    continue

            if not output_path.exists():
                continue

            links = tuple(_read_archive_links_from_html(output_path))
            if not links:
                print(f"Останавливаем {year}: на странице {page} нет XML-ссылок архива")
                break

            if links in previous_link_sets:
                print(f"Останавливаем {year}: страница {page} повторяет предыдущие XML архива")
                break

            previous_link_sets.add(links)
            pages_with_links.append(output_path)

            if _is_last_archive_page(output_path):
                print(f"Останавливаем {year}: страница {page} последняя в архиве")
                break

    print(f"Страниц архива Росказны с XML-ссылками: {len(set(pages_with_links))}")
    print(f"Из них скачано в этом прогоне: {fetched_count}")
    return sorted(set(pages_with_links)), fetched_count


def collect_roskazna_xml_links(
    pages_dir: Path = PAGES_DIR,
    links_file: Path = LINKS_FILE,
) -> list[str]:
    """Собирает XML-ссылки Росказны из HTML-страниц и txt-файла"""
    links: list[str] = []

    if pages_dir.exists():
        page_paths = sorted(pages_dir.glob("*.html"))
        for path in page_paths:
            links.extend(_read_archive_links_from_html(path))

        if page_paths:
            links.extend(_read_current_links_from_html(page_paths[-1]))

    links.extend(_read_links_from_txt(links_file))

    unique_links: list[str] = []
    seen_links: set[str] = set()
    for link in links:
        if link in seen_links:
            continue
        seen_links.add(link)
        unique_links.append(link)

    return unique_links


def download_roskazna_xml_files(
    links: list[str],
    raw_dir: Path = RAW_DIR,
    force: bool = False,
) -> tuple[list[Path], int]:
    """Скачивает XML-файлы Росказны по списку ссылок.

    Возвращает (локальные файлы по ссылкам, сколько скачано в этом прогоне).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []
    fetched_count = 0
    skipped_count = 0
    failed_count = 0

    for link in links:
        output_path = _xml_output_path(link, raw_dir)
        if output_path.exists() and not force:
            skipped_count += 1
            downloaded_files.append(output_path)
            continue

        try:
            _download_roskazna_file(link, output_path)
        except RoskaznaTlsError:
            raise
        except Exception as error:
            failed_count += 1
            print(f"Не удалось скачать XML Росказны: {link}")
            print(f"Причина: {error}")
            continue

        downloaded_files.append(output_path)
        fetched_count += 1

    print(f"Найдено XML-ссылок Росказны: {len(links)}")
    print(f"Пропущено уже скачанных XML: {skipped_count}")
    print(f"Не скачано XML из-за ошибок: {failed_count}")
    print(f"Скачано новых XML в этом прогоне: {fetched_count}")

    return sorted(set(downloaded_files)), fetched_count


def _local_xml_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Возвращает локальные XML-файлы Росказны"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".xml"
    )


def _download_pages_by_year(years: list[int]) -> tuple[int, bool]:
    """Качает HTML-архив: прошлые годы из кеша (force=False), текущий год — заново.

    Возвращает (скачано страниц текущего года, запрашивался ли текущий год).
    Прошлые годы намеренно не влияют на результат: они и должны браться из кеша.
    """
    today_year = date.today().year
    past_years = [year for year in years if year < today_year]
    current_years = [year for year in years if year >= today_year]

    if past_years:
        download_roskazna_html_pages(years=past_years, force=False)

    if not current_years:
        return 0, False

    # текущий год перекачиваем, чтобы увидеть свежие депозиты
    _, fetched = download_roskazna_html_pages(years=current_years, force=True)
    return fetched, True


def prepare_roskazna_treasury_deposits(
    raw_dir: Path = RAW_DIR,
    *,
    update_pages: bool = True,
    years: list[int] | None = None,
) -> list[Path]:
    """Готовит raw-файлы Росказны для пайплайна.

    Сам скачивает HTML-архив, а не полагается на ранее закешированные страницы.

    Успех определяется результатом ТЕКУЩЕГО прогона, а не наличием файлов на
    диске. Раньше проверялось `not any(PAGES_DIR.glob("*.html"))`, поэтому при
    непустом кеше сетевой сбой не отличался от нормального обновления: шаг M5
    отчитывался «ok» на данных двухмесячной давности. Теперь при живом кеше и
    мёртвой сети шаг падает с понятной причиной.
    """
    if years is None:
        years = list(range(DEFAULT_START_YEAR, date.today().year + 1))

    if update_pages:
        fetched_pages, current_year_requested = _download_pages_by_year(years)
        if current_year_requested and fetched_pages == 0:
            raise RuntimeError(
                "Источник Росказны не отдал ни одной страницы архива за текущий год. "
                f"На диске могут лежать старые страницы — они НЕ считаются успехом. "
                f"Проверьте доступ к {SOURCE_URL}"
            )

    links = collect_roskazna_xml_links()
    if links:
        download_roskazna_xml_files(links, raw_dir)

    files = _local_xml_files(raw_dir)

    if not files:
        raise FileNotFoundError(
            "Не найдены XML-файлы Росказны: не удалось скачать HTML-архив и собрать "
            f"XML-ссылки. Проверьте доступ к источнику: {SOURCE_URL}"
        )

    return files


def _parse_years(value: str | None) -> list[int]:
    """Парсит список лет из аргумента командной строки"""
    if value is None:
        return list(range(DEFAULT_START_YEAR, date.today().year + 1))

    years: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            years.extend(range(int(start_text), int(end_text) + 1))
        else:
            years.append(int(text))

    return sorted(set(years))


def main() -> None:
    """Скачивает и проверяет raw-файлы Росказны"""
    argument_parser = argparse.ArgumentParser(
        description="Скачивает XML Росказны по сохраненным HTML-страницам архива"
    )
    argument_parser.add_argument(
        "--no-update-pages",
        action="store_true",
        help="Не скачивать HTML-страницы архива, использовать только локальные HTML/txt",
    )
    argument_parser.add_argument(
        "--years",
        help="Годы для скачивания HTML, например 2024 или 2021-2026",
    )
    argument_parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_YEAR,
        help="Максимум страниц архива на один год",
    )
    argument_parser.add_argument(
        "--force-pages",
        action="store_true",
        help="Перекачивать HTML-страницы, даже если они уже есть",
    )
    argument_parser.add_argument(
        "--force-xml",
        action="store_true",
        help="Перекачивать XML-файлы, даже если они уже есть",
    )
    args = argument_parser.parse_args()

    bundle = resolve_ca_bundle()
    print(f"CA-бандл для Росказны: {bundle if bundle else 'не задан (системное хранилище)'}")
    if resolve_pinned_fingerprint() is not None:
        print(f"Режим пиннинга включён через {PINNED_CERT_ENV_VAR}")

    if not args.no_update_pages:
        download_roskazna_html_pages(
            years=_parse_years(args.years),
            max_pages_per_year=args.max_pages,
            force=args.force_pages,
        )

    links = collect_roskazna_xml_links()
    if links:
        download_roskazna_xml_files(links, force=args.force_xml)

    files = _local_xml_files()
    if not files:
        raise FileNotFoundError(
            "Не найдены XML-файлы Росказны и XML-ссылки для скачивания"
        )

    print(f"Найдено XML-файлов Росказны: {len(files)}")


if __name__ == "__main__":
    main()
