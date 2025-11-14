import argparse
import asyncio
import logging
import logfire

from hostile_copilot.config import load_config, OmegaConfig
from hostile_copilot.client.app import HostileCoPilotApp
from hostile_copilot.client.tasks.extract_locations import ExtractLocationsTask
from hostile_copilot.client.tasks.macro import MacroTask, MacroStepT
import ast

async def run_search_gw(query: str, app: HostileCoPilotApp) -> int:
    # Call the tool and print results
    results = await app._tool_search_gravity_well_locations(query)
    for item in results:
        print(item)

    return 0


async def run_loc_extract(filename: str, app: HostileCoPilotApp) -> int:
    task = ExtractLocationsTask(app._config, app._app_config, app._location_provider, filename)
    await task.run()

    for loc in (task.locations or []):
        print(loc)

    return 0

async def run_macro(macro_str: str | None, app: HostileCoPilotApp) -> int:
    task = MacroTask(app._config, app._keyboard)

    async def async_input(prompt: str) -> str:
        return await asyncio.to_thread(input, prompt)

    if macro_str is not None:
        macro = _parse_macro(macro_str)
        task.set_macro(macro)
        await task.run()
    else:
        # Run repl
        while True:
            try:
                try:
                    line = await async_input("Macro: ")
                    macro_str = line.strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not macro_str:
                    continue

                if macro_str.lower() == "exit" or macro_str.lower() == "quit":
                    break

                macro = _parse_macro(macro_str)
                print(macro)
                task.set_macro(macro)
                await task.run()
            except Exception as e:
                print(f"Error running macro: {e}")
                import traceback
                traceback.print_exc()
            finally:
                await asyncio.sleep(0)
    return 0

def _parse_macro(macro: str) -> list[MacroStepT]:
    """
    Parse a macro string into a list of MacroStepT.

    Supported forms (comma-separated at top-level):
      - click(100,200)
      - moveTo(300,400)
      - vkbd:press('enter')
      - vkbd:sequence("hello world")
      - vkbd:sleep(0.5)
      - click(100,200, clicks=2)

    Returns list of tuples: (action, args) or (action, args, kwargs)
    where action is the function name like 'click', 'moveTo', 'vkbd:press'.
    """

    def top_level_split(s: str, delimiter: str = ',') -> list[str]:
        parts: list[str] = []
        buf: list[str] = []
        depth = 0
        in_quote: str | None = None
        i = 0
        while i < len(s):
            ch = s[i]
            if in_quote:
                buf.append(ch)
                if ch == in_quote:
                    in_quote = None
                elif ch == '\\':
                    # skip escaped next char within quotes
                    i += 1
                    if i < len(s):
                        buf.append(s[i])
                i += 1
                continue
            if ch in ('"', "'"):
                in_quote = ch
                buf.append(ch)
            elif ch in '([{':
                depth += 1
                buf.append(ch)
            elif ch in ')]}':
                depth = max(0, depth - 1)
                buf.append(ch)
            elif ch == delimiter and depth == 0:
                part = ''.join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
            else:
                buf.append(ch)
            i += 1
        tail = ''.join(buf).strip()
        if tail:
            parts.append(tail)
        return parts

    def parse_value(token: str):
        token = token.strip()
        try:
            return ast.literal_eval(token)
        except Exception:
            # try numeric
            try:
                if token.lower().startswith(('0x','-0x')):
                    return int(token, 16)
                if '.' in token:
                    return float(token)
                return int(token)
            except Exception:
                # bare word -> string
                return token

    def parse_call(call_str: str) -> MacroStepT:
        call_str = call_str.strip()
        if not call_str:
            raise ValueError('Empty call')
        if '(' not in call_str:
            action = call_str
            args = None
            return (action, args)
        idx = call_str.find('(')
        action = call_str[:idx].strip()
        if not call_str.endswith(')'):
            raise ValueError(f"Malformed call: {call_str}")
        arg_str = call_str[idx+1:-1].strip()
        if not arg_str:
            return (action, None)
        # split args/kwargs at top-level
        items = top_level_split(arg_str, ',')
        pos_vals: list = []
        kwargs: dict = {}
        for item in items:
            if '=' in item and not item.strip().startswith(('"', "'")):
                k, v = item.split('=', 1)
                kwargs[k.strip()] = parse_value(v)
            else:
                pos_vals.append(parse_value(item))
        if len(pos_vals) == 0:
            args = None
        elif len(pos_vals) == 1:
            args = pos_vals[0]
        else:
            args = tuple(pos_vals)
        if kwargs:
            return (action, args, kwargs)
        else:
            return (action, args)

    # split into calls at top-level commas
    calls = top_level_split(macro)
    steps: list[MacroStepT] = []
    for c in calls:
        if not c:
            continue
        steps.append(parse_call(c))
    return steps

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HostileCoPilot tools tester")
    parser.add_argument("--config", "-c", default=None, help="Path to configuration file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search_gw subcommand
    p_search = subparsers.add_parser("search_gw", help="Search gravity well locations")
    p_search.add_argument("query", help="Search query for gravity well locations")

    # loc_extract subcommand
    p_extract = subparsers.add_parser("loc_extract", help="Extract locations from a file")
    p_extract.add_argument("filename", help="Path to image/text file containing locations")

    # macro subcommand
    p_macro = subparsers.add_parser("macro", help="Run a macro")
    p_macro.add_argument("macro", nargs="*", help="Macro to run")

    return parser

async def load_app(config: OmegaConfig) -> HostileCoPilotApp:
    app = HostileCoPilotApp(config)
    await app.initialize(listen=False)
    return app

async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logfire_instance = logfire.configure(
        service_name="hostile_copilot",
        environment="development",
        console=False
    )

    logfire_instance.instrument_pydantic_ai()
    logfire_instance.instrument_requests()

    # Load config
    config: OmegaConfig = load_config(args.config)
    app = await load_app(config)


    if args.command == "search_gw":
        return await run_search_gw(args.query, app)
    elif args.command == "loc_extract":
        return await run_loc_extract(args.filename, app)
    elif args.command == "macro":
        macro_str = " ".join(args.macro) if args.macro else None
        return await run_macro(macro_str, app)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
