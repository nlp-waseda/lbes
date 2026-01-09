import ast
import re
from collections import Counter
from fractions import Fraction

_PATTERN = re.compile(r"^.*<think>.+</think>.*<answer>(.+?)</answer>.*$", re.DOTALL)

_ALLOWED_EXPR_CHARS = re.compile(r"^[0-9+\-*/() ]+$")


def _extract_answer(text: str) -> str | None:
    match = _PATTERN.match(text)
    if (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and text.count("<answer>") == 1
        and text.count("</answer>") == 1
        and match is not None
    ):
        return match.group(1).strip()
    return None


def _is_supported_ast(node: ast.AST) -> bool:
    """Whitelist-only AST check for safe evaluation."""
    match node:
        case ast.Expression(body=body):
            return _is_supported_ast(body)
        case ast.BinOp(left=left, op=op, right=right):
            return (
                isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div))
                and _is_supported_ast(left)
                and _is_supported_ast(right)
            )
        case ast.UnaryOp(op=op, operand=operand):
            return isinstance(op, (ast.UAdd, ast.USub)) and _is_supported_ast(operand)
        case ast.Constant(value=value):
            return isinstance(value, int) and not isinstance(value, bool)
        case _:
            return False


def _collect_int_literals(node: ast.AST, out: list[int]) -> None:
    if isinstance(node, ast.Expression):
        _collect_int_literals(node.body, out)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            out.append(int(node.value))
        return
    if isinstance(node, ast.UnaryOp):
        _collect_int_literals(node.operand, out)
        return
    if isinstance(node, ast.BinOp):
        _collect_int_literals(node.left, out)
        _collect_int_literals(node.right, out)
        return


def _eval_fraction(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_fraction(node.body)

    if isinstance(node, ast.Constant):
        if not (isinstance(node.value, int) and not isinstance(node.value, bool)):
            raise ValueError("Only integer constants are allowed")
        return Fraction(int(node.value), 1)

    if isinstance(node, ast.UnaryOp):
        v = _eval_fraction(node.operand)
        if isinstance(node.op, ast.UAdd):
            return v
        if isinstance(node.op, ast.USub):
            return -v
        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_fraction(node.left)
        right = _eval_fraction(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        raise ValueError("Unsupported binary operator")

    raise ValueError("Unsupported expression")


def _parse_and_validate_expression(
    expr: str, numbers: list[int] | None
) -> ast.AST | None:
    """Parse and validate expression; returns AST on success, else None."""
    if not expr or _ALLOWED_EXPR_CHARS.fullmatch(expr) is None:
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    if not _is_supported_ast(tree):
        return None

    if numbers is not None:
        literals: list[int] = []
        _collect_int_literals(tree, literals)
        if Counter(literals) != Counter(numbers):
            return None

    return tree


# "reward" is necessary
def compute_score(
    completion_text: str,
    *,
    target: int,
    numbers: list[int] | None = None,
) -> dict[str, float]:
    expr = _extract_answer(completion_text)
    if expr is None:
        return {"reward": 0.0, "format": 0.0, "accuracy": 0.0}

    tree = _parse_and_validate_expression(expr, numbers)
    if tree is None:
        return {"reward": 0.0, "format": 0.0, "accuracy": 0.0}

    try:
        value = _eval_fraction(tree)
    except (ValueError, ZeroDivisionError):
        return {"reward": 0.0, "format": 0.0, "accuracy": 0.0}

    is_correct = value == Fraction(int(target), 1)
    if is_correct:
        return {"reward": 1.1, "format": 0.1, "accuracy": 1.0}
    return {"reward": 0.1, "format": 0.1, "accuracy": 0.0}
