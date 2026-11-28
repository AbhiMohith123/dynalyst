"""What the notebook shows: math any viewer renders, and Plotly figures that never get lost."""
import plotly.graph_objects as go

from dynalyst import notebook


def test_math_delimiters_become_dollars_outside_code():
    text = "Fast: \(AP, AK\). `\(code\)`\n\[\n\dot{x} = k\,x\n\]\nend"
    out = notebook.markdown_math(text)
    assert "Fast: $AP, AK$." in out
    assert "`\(code\)`" in out                      # code is left as written
    assert "\n$$\n\dot{x} = k\,x\n$$\n" in out


def test_a_figure_is_shown_as_html(monkeypatch):
    shown = []
    monkeypatch.setattr(notebook, "display", lambda x: shown.append(x))
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    notebook.show_figure(fig)
    assert len(shown) == 1 and "plotly" in shown[0].data.lower()


def test_an_answer_displays_its_plots_itself(monkeypatch, tmp_path):
    from dynalyst import notebook
    shown = []
    monkeypatch.setattr(notebook, "display", lambda x: shown.append(x))
    png = tmp_path / "a.png"
    png.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
    notebook.show_answer(f"Before ![a plot]({png}) after ![gone](outputs/not_there.png) end")
    kinds = [type(x).__name__ for x in shown]
    assert kinds == ["Markdown", "Image", "Markdown", "Markdown", "Markdown"]
    assert "plot not found" in shown[3].data and notebook._same_file(png) in notebook._SHOWN


def test_a_path_with_spaces_and_brackets_and_a_repeated_link_show_once(monkeypatch, tmp_path):
    from dynalyst import notebook
    shown = []
    monkeypatch.setattr(notebook, "display", lambda x: shown.append(x))
    folder = tmp_path / "agent v3.1 (2)"
    folder.mkdir()
    png = folder / "a.png"
    png.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
    notebook.show_answer(f"![plot [step 1]]({png}) and again ![plot]({png})")
    assert [type(x).__name__ for x in shown] == ["Image", "Markdown"]
