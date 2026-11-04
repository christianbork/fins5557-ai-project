"""The DDF map in src/ddf.py covers the whole codebase, once."""

from src.config import ROOT
from src.ddf import DDF_STATIONS, station_of


def test_the_four_course_stations_in_order():
    names = [s["name"] for s in DDF_STATIONS]
    assert names == ["ETL", "Feature Engineering", "Model Design", "Implementation"]
    assert [s["station"] for s in DDF_STATIONS] == [1, 2, 3, 4]


def test_every_module_and_page_has_exactly_one_station():
    files = ["app.py"] + [
        p.relative_to(ROOT).as_posix()
        for folder in ("src", "pages")
        for p in sorted((ROOT / folder).glob("*.py"))
        if p.name not in ("__init__.py", "config.py")  # config: settings shared by every station
    ]
    listed = [m for s in DDF_STATIONS for m in s["modules"]]
    assert sorted(listed) == sorted(set(listed)), "a module is listed under two stations"
    assert sorted(files) == sorted(listed)
    assert station_of("src/llm.py") == 4 and station_of("src/data.py") == 1


def test_module_docstrings_name_their_station():
    for station in DDF_STATIONS:
        for module in station["modules"]:
            first_line = (ROOT / module).read_text(encoding="utf-8").lstrip().splitlines()[0]
            assert f"Station {station['station']}" in first_line, module
