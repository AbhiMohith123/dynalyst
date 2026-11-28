"""
A small ipywidgets chat box for driving the bifurcation agent from a text
box + button instead of retyping ask("...") in a new cell every time.

Usage, in a notebook cell:

    from dynalyst.chat_ui import launch_chat_ui
    launch_chat_ui()                   # the agent (planner -> sub-agents -> reviewer)

The conversation so far is passed to the agent as history. For the older
single-purpose agent: launch_chat_ui(agent="research", show_agent_picker=True).

That displays the widget once; leave that cell as-is and keep typing new
queries into the box -- you don't need to re-run the cell or open a new
one per query. The conversation (your queries + the agent's replies +
every plot/table it produces) accumulates in the output area below the
box, like a real chat transcript, rather than being replaced each turn.
"""
import traceback
import ipywidgets as widgets
from IPython.display import display

from dynalyst.notebook import AGENTS, ask


def launch_chat_ui(show_csv=True, show_xlsx=True, agent="unified", show_agent_picker=False):
    """Builds and displays the chat widget. Returns the container widget
    (rarely needed -- mainly so you can re-display() it if the cell output
    ever gets cleared)."""
    query_box = widgets.Textarea(
        placeholder='Ask something, e.g. "Run a bifurcation analysis of <model file> and '
                    'tell me where the folds are"',
        layout=widgets.Layout(width='100%', height='70px'),
    )
    send_button = widgets.Button(description='Send', button_style='primary',
                                  icon='paper-plane')
    clear_button = widgets.Button(description='Clear chat', icon='trash')
    agent_picker = widgets.Dropdown(options=list(AGENTS), value=agent, description='Agent:',
                                    layout=widgets.Layout(width='220px'))
    history = []  # conversation_history for the unified/research agents
    status_label = widgets.Label(value='')
    follow_up_box = widgets.HBox([])  # one button per suggested follow-up
    output_area = widgets.Output(layout=widgets.Layout(
        border='1px solid #ddd', padding='8px', max_height='600px', overflow='auto'))

    def _set_busy(busy):
        send_button.disabled = busy
        clear_button.disabled = busy
        query_box.disabled = busy
        status_label.value = 'Running...  (this can take a while for LHS/AUTO scans)' if busy else ''

    def show_follow_ups():
        from dynalyst import notebook
        rec = notebook.LAST_RUN
        buttons = []
        for item in (rec.follow_ups if rec is not None else []):
            btn = widgets.Button(description=item["question"][:60] + ("..." if len(item["question"]) > 60 else ""),
                                 tooltip=item["query"], layout=widgets.Layout(width="auto"))
            btn.on_click(lambda _b, q=item["query"]: send(q))
            buttons.append(btn)
        follow_up_box.children = buttons

    def on_send_click(b):
        send(query_box.value.strip())

    def send(query):
        if not query:
            return
        follow_up_box.children = []
        query_box.value = ''
        _set_busy(True)
        with output_area:
            print(f">>> {query}\n")
            try:
                chosen = agent_picker.value
                answer = ask(query, show_csv=show_csv, show_xlsx=show_xlsx, agent=chosen,
                             history=list(history))
                if isinstance(answer, str):
                    history.extend([{"role": "user", "content": query},
                                    {"role": "assistant", "content": answer}])
                    del history[:-20]
            except Exception:
                # Surface the real traceback in the chat transcript rather
                # than letting the button handler swallow it silently --
                # an ipywidgets on_click callback that raises just prints
                # to stderr in some frontends and the chat box looks like
                # it silently did nothing.
                print("--- error while handling this query ---")
                traceback.print_exc()
            print("\n" + "-" * 80 + "\n")
        _set_busy(False)
        if agent_picker.value == "unified":
            show_follow_ups()

    def on_clear_click(b):
        from dynalyst import notebook
        from dynalyst.agents.unified_agent import reset_agents
        from dynalyst.orchestration.state import SessionState
        reset_agents()
        SessionState().clear()      # a cleared chat starts with no memory of earlier questions
        notebook.LAST_RUN = None
        follow_up_box.children = []
        query_box.value = ''
        output_area.clear_output()
        history.clear()

    send_button.on_click(on_send_click)
    clear_button.on_click(on_clear_click)

    # Ctrl+Enter / Shift+Enter don't reliably submit a Textarea across
    # ipywidgets/frontend versions, so Send is the one guaranteed path --
    # kept as a single explicit button rather than promising a keyboard
    # shortcut that might silently not fire for you.
    controls = widgets.HBox([send_button, clear_button]
                            + ([agent_picker] if show_agent_picker else []) + [status_label])
    chat_box = widgets.VBox([query_box, controls, follow_up_box, output_area])
    display(chat_box)
    return chat_box
