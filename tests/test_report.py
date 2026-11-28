"""save_report: a run record as HTML and PDF with its plots copied beside it. No network."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynalyst import config


def test_a_run_becomes_an_html_and_pdf_report_with_its_plot(tmp_path, monkeypatch):
    import pytest
    pymupdf = pytest.importorskip("pymupdf")
    from dynalyst.orchestration.agent import RunRecord
    from dynalyst.report import save_report
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    png = tmp_path / "diagram.png"
    plt.plot([0, 1], [0, 1]); plt.savefig(png); plt.close("all")
    rec = RunRecord(query="Bifurcation of model.xml in k1", model={"orchestrator": "test-model"})
    rec.final_answer = (f"Folds at **k1 = 2.31** and 4.87 [step 1].\n\n| point | k1 |\n|---|---|\n"
                        f"| LP | 2.31 |\n\n![diagram]({png})")
    rec.steps = [{"tool": "run_bifurcation_analysis", "status": "ok",
                  "arguments": {"model_name_or_path": "model.xml", "param_name": "k1"},
                  "reviewer": {"verdict": "accept"}}]
    out = save_report("demo", run=rec)
    page = open(out["html"], encoding="utf-8").read()
    assert "k1 = 2.31" in page and "run_bifurcation_analysis" in page and "demo_files/01_diagram.png" in page
    assert (tmp_path / "reports" / "demo_files" / "01_diagram.png").is_file()
    doc = pymupdf.open(out["pdf"])
    text = "".join(p.get_text() for p in doc)
    assert doc.page_count >= 1 and "2.31" in text and "Bifurcation of model.xml" in text
    assert sum(len(p.get_images()) for p in doc) >= 1          # the plot is drawn in the PDF
