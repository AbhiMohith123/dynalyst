"""AUTO-07p on Windows, Linux and macOS: one library per platform (lib/<platform>/),
objects compiled elsewhere are refused with a rebuild message, and AUTO's runner keeps
Windows paths intact. No AUTO build is needed for these tests."""
import platform
import sys

import pytest

from dynalyst import auto_env

MACHO_ARM64 = b"\xcf\xfa\xed\xfe" + (0x0100000C).to_bytes(4, "little")
ELF_X86_64 = b"\x7fELF" + bytes(14) + (0x3E).to_bytes(2, "little")
COFF_X86_64 = (0x8664).to_bytes(2, "little") + bytes(18)


def _native_header():
    cpu = auto_env._cpu(platform.machine())
    if sys.platform.startswith("darwin"):
        return b"\xcf\xfa\xed\xfe" + ({"arm64": 0x0100000C}.get(cpu, 0x01000007)).to_bytes(4, "little")
    if sys.platform.startswith("linux"):
        return b"\x7fELF" + bytes(14) + ({"arm64": 0xB7}.get(cpu, 0x3E)).to_bytes(2, "little")
    return ({"arm64": 0xAA64}.get(cpu, 0x8664)).to_bytes(2, "little") + bytes(18)


def _tree(tmp_path, header=None):
    (tmp_path / "python" / "auto").mkdir(parents=True)
    (tmp_path / "lib").mkdir()
    if header is not None:
        (tmp_path / "lib" / "ae.o").write_bytes(header + bytes(40))
    return tmp_path


@pytest.mark.parametrize("header,plat,cpu", [(MACHO_ARM64, "darwin", "arm64"),
                                             (ELF_X86_64, "linux", "x86_64"),
                                             (COFF_X86_64, "win32", "x86_64")])
def test_object_headers_name_their_platform(tmp_path, header, plat, cpu):
    p = tmp_path / "x.o"
    p.write_bytes(header + bytes(40))
    assert auto_env.object_platform(p)[:2] == (plat, cpu)


def test_the_build_for_this_platform_is_used(tmp_path):
    root = _tree(tmp_path, MACHO_ARM64 if not sys.platform.startswith("darwin") else ELF_X86_64)
    own = auto_env.lib_dir(root)
    own.mkdir(parents=True)
    (own / "ae.o").write_bytes(_native_header())
    (own / auto_env.BUILD_STAMP).write_text('{"host": "%s"}' % auto_env.this_host())
    assert auto_env.find_lib_dir(root) == own          # not the foreign lib/*.o


def test_a_library_built_on_another_machine_is_refused(tmp_path, monkeypatch):
    """A project copied from another machine brings that machine's build along: it
    must be rebuilt here, unless the machines share one toolchain."""
    import json
    root = _tree(tmp_path)
    own = auto_env.lib_dir(root)
    own.mkdir(parents=True)
    (own / "ae.o").write_bytes(_native_header())
    (own / auto_env.BUILD_STAMP).write_text(json.dumps({"host": "someone-elses-pc"}))
    with pytest.raises(RuntimeError) as e:
        auto_env.find_lib_dir(root)
    assert "someone-elses-pc" in str(e.value) and "--force" in str(e.value)
    assert auto_env.build_failure(str(e.value))
    monkeypatch.setenv("DYNALYST_AUTO_SHARED_BUILD", "1")
    assert auto_env.find_lib_dir(root) == own
    (own / auto_env.BUILD_STAMP).write_text(json.dumps({"host": auto_env.this_host()}))
    monkeypatch.delenv("DYNALYST_AUTO_SHARED_BUILD")
    assert auto_env.find_lib_dir(root) == own


def test_an_unfinished_build_is_not_used(tmp_path):
    root = _tree(tmp_path)
    own = auto_env.lib_dir(root)
    own.mkdir(parents=True)
    (own / "ae.o").write_bytes(_native_header())        # no BUILD_INFO.json: interrupted
    with pytest.raises(RuntimeError, match="has not been built"):
        auto_env.find_lib_dir(root)


def test_objects_from_another_machine_are_refused_as_a_setup_problem(tmp_path):
    foreign = ELF_X86_64 if sys.platform.startswith("darwin") else MACHO_ARM64
    root = _tree(tmp_path, foreign)
    with pytest.raises(RuntimeError) as e:
        auto_env.find_lib_dir(root)
    assert "cannot be linked here" in str(e.value) and "build_auto.py" in str(e.value)
    assert auto_env.build_failure(str(e.value))         # reported as AUTO unavailable


def test_an_older_in_place_build_for_this_machine_still_works(tmp_path):
    root = _tree(tmp_path, _native_header())
    assert auto_env.find_lib_dir(root) == root / "lib"


def test_windows_paths_survive_autos_command_splitting(monkeypatch):
    if not (auto_env.auto_dir() / "python" / "auto").is_dir():
        pytest.skip("vendor/auto-07p is not unpacked (python scripts/build_auto.py unpacks it)")
    sys.path.insert(0, str(auto_env.auto_dir() / "python"))
    from auto import runAUTO
    monkeypatch.setattr(runAUTO.os, "name", "nt")
    cmd = r"gfortran -g fold.o -o fold.exe E:\proj\auto\lib\ae.o C:\My\ Files\x.o"
    assert runAUTO._split(cmd) == ["gfortran", "-g", "fold.o", "-o", "fold.exe",
                                   "E:/proj/auto/lib/ae.o", "C:/My Files/x.o"]
    assert runAUTO._split(r".\fold.exe") == ["./fold.exe"]
