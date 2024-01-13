import sys
from collections import OrderedDict
from collections import defaultdict
from copy import deepcopy

import sympy
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

from data.datasetgenerator import save_result_as_gzipped_json
from data.tree import Node

all_symbols = [chr(i) for i in range(97, 98)]

# Name, (preorder, inorder, postorder), child_properties
non_terminals = [('Add', '+', ('left', 'right')),
                 ('Subtract', '-', ('left', 'right')),
                 ('Multiply', '*', ('left', 'right')),
                 ]

print_instructions = {s[0]: s[1] for s in non_terminals}


def generate_all_trees(current_tree, max_tree_size=13):
    if len(current_tree) > max_tree_size + 1:
        return

    empty_positions = [(n, p) for n in current_tree for p in n.properties if len(n[p]) == 0]
    if len(current_tree) + len(empty_positions) > max_tree_size + 1:
        return
    if len(empty_positions) == 0:
        yield current_tree
        return

    for symbol in all_symbols:
        tree_copy = deepcopy(current_tree)
        node, property = next((n, p) for n in tree_copy for p in n.properties if len(n[p]) == 0)
        child = Node(symbol, (), parent=node)
        node.set_children_for_property(property, (child,))
        yield from generate_all_trees(tree_copy)

    for non_terminal_name, _, properties in non_terminals:
        tree_copy = deepcopy(current_tree)
        node, property = next((n, p) for n in tree_copy for p in n.properties if len(n[p]) == 0)
        child = Node(non_terminal_name, properties, parent=node)
        node.set_children_for_property(property, (child,))
        yield from generate_all_trees(tree_copy)


def to_token_sequence(node: Node, current_tokens: list) -> list:
    if node.name not in print_instructions and node.name != 'Start':
        current_tokens.append(node.name)
        return current_tokens
    elif node.name == 'Start':
        return to_token_sequence(node['child'][0], current_tokens)
    symbol_token = print_instructions[node.name]
    node_properties = node.properties

    is_symbol = node[node_properties[0]][0].name not in print_instructions
    if not is_symbol: current_tokens.append("(")

    assert len(node_properties) == 2
    to_token_sequence(node[node_properties[0]][0], current_tokens)

    if not is_symbol: current_tokens.append(")")
    current_tokens.append(symbol_token)

    is_symbol = node[node_properties[1]][0].name not in print_instructions
    if not is_symbol: current_tokens.append("(")
    to_token_sequence(node[node_properties[1]][0], current_tokens)
    if not is_symbol: current_tokens.append(")")
    return current_tokens


def convert_to_dict(node: Node) -> dict:
    children = OrderedDict()
    for node_property in node.properties:
        children[node_property] = convert_to_dict(node[node_property][0])
    simplified = str(sympy.expand(parse_expr(''.join(to_token_sequence(node, [])))))
    if len(children) > 0:
        return dict(Name=node.name, Children=children, Symbol=simplified)
    else:
        return dict(Name=node.name, Symbol=simplified)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage <outputFilenamePrefix>")
        sys.exit(-1)
    synthesized_expressions = defaultdict(list)
    num_times_returned_no_tree = 0
    num_times_returned_duplicate = 0


    def print_stats():
        print(
            "Generated None %s times, duplicates %s times" % (num_times_returned_no_tree, num_times_returned_duplicate))
        print("Generated %s unique expressions (%s in total)" % (
        len(synthesized_expressions), sum(len(e) for e in synthesized_expressions.values())))


    tree_generator = generate_all_trees(Node('Start', ('child',)))
    for i, tree in tqdm(enumerate(tree_generator)):
        if i % 5000 == 4999:
            print_stats()

        tokens = to_token_sequence(tree, [])
        expression = sympy.expand(parse_expr(''.join(tokens)))
        all_elements = synthesized_expressions[str(expression)]
        exists = False
        for element in all_elements:
            if element[0] == tokens:
                exists = True
                break
        if not exists:
            all_elements.append((tokens, tree))
        else:
            num_times_returned_duplicate += 1
    print_stats()


    def save_to_json_gz(data, filename):
        converted_to_standard_format = {}
        for n, all_expressions in data.items():
            expression_dicts = [dict(Tokens=expr[0], Tree=convert_to_dict(expr[1])) for expr in all_expressions]
            converted_to_standard_format[n] = dict(Original=expression_dicts[0], Noise=expression_dicts[1:])

        save_result_as_gzipped_json(filename, converted_to_standard_format)


    save_to_json_gz(synthesized_expressions, sys.argv[1] + ".json.gz")
