import sys
from rich.console import Console
from rich.panel import Panel
from src.pipeline import SupportPipeline

console = Console()


def run_cli():
    """Boots the interactive QTrade AI Support terminal interface."""
    console.print(
        Panel.fit(
            "[bold]Welcome to the QTrade AI Support Assistant[/]\n"
            "Type your support inquiry below, or type [bold red]'exit'[/] to quit.",
            title="QTrade AI Support CLI",
            border_style="cyan",
        )
    )

    with console.status("[bold cyan]Initializing knowledge base pipeline...[/]"):
        try:
            pipeline = SupportPipeline()
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]Fatal Error initializing pipeline:[/] {str(e)}\n"
                    "Please check your API key in the .env file.",
                    title="Startup Failure",
                    border_style="red",
                )
            )
            sys.exit(1)

    chat_history = []

    while True:
        try:
            user_input = console.input("\n[bold cyan]Customer:[/] ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold green]Thank you for contacting QTrade. Goodbye![/]")
                break

            with console.status("[bold green]Searching help documents & generating answer...[/]"):
                response = pipeline.process_query(user_input, chat_history)

            # Update history to maintain conversational memory
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response.answer})

            # STATE 1: ESCALATION TRIGGERED
            if response.escalation.should_escalate:
                summary = (
                    response.escalation.handoff_summary
                    or response.escalation.reason
                    or "Escalated without additional details."
                )
                panel_content = (
                    f"[bold red]{response.answer}[/]\n\n"
                    f"[dim italic]Handoff Summary: {summary}[/]"
                )
                console.print(
                    Panel(
                        panel_content,
                        title="ESCALATION TO HUMAN AGENT",
                        border_style="red",
                    )
                )

            # STATE 2: UNKNOWN INQUIRY ("I don't know")
            elif response.answer.lower() == "i don't know.":
                console.print(
                    Panel(
                        "[bold yellow]I don't know.[/]\n\n"
                        "[dim italic]This inquiry is outside the scope of our help documentation.[/]",
                        title="Unknown Inquiry",
                        border_style="yellow",
                    )
                )

            # STATE 3: GROUNDED CITED ANSWER
            else:
                citation_box = ""
                if response.citation:
                    citation_box = f"\n\n[dim cyan]Grounded Source:[/] [bold]{response.citation.source_doc}[/]"

                console.print(
                    Panel(
                        response.answer + citation_box,
                        title="QTrade AI Assistant",
                        border_style="green",
                    )
                )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Session terminated. Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[bold red]Unexpected runtime error:[/] {str(e)}")