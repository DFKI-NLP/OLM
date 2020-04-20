import numpy as np


def visualize_relevances(input_instances, relevances, labels_true = None, labels_pred = None, font_size=5):
    def rgba(relevance):
        if relevance >= 0:
            return f"rgba(255, 0, 0, {relevance})"
        else:
            return f"rgba(0, 0, 255, {abs(relevance)})"

    def color(relevance):
        if relevance > 0.8:
            return "white"
        else:
            return "black"

    visualized_inputs = []
    for i, instance in enumerate(input_instances):
        for field_name, token_field in instance.token_fields.items():
            tokens_relevance = relevances[instance.id]
            max_relevance = max(np.abs(list(tokens_relevance.values())))
            norm_tokens_relevance = {idx: r / max_relevance for idx, r in tokens_relevance.items()}

            html_tokens = []
            for idx, token in enumerate(token_field.tokens):
                relevance = norm_tokens_relevance[(field_name, idx)]
                html_token = f'<span style="color:{color(relevance)}; background-color:{rgba(relevance)};">{token}</span>'
                html_tokens.append(html_token)

            if labels_true is not None:
                correct = ""
                if labels_pred is not None:
                    correct = "&#10004;" if labels_true[i] == labels_pred[i] else "&#10006;"

                prefix = '<span style="color:black; background-color:rgba(255, 255, 0, 0.6);">' \
                    + f'{labels_true[i]} {correct} {max_relevance:.2f}</span>:   '
            else:
                prefix = ""

            visualized_input = f'<font size="{font_size}">' + prefix + " ".join(html_tokens) + '</font>'

            visualized_inputs.append(visualized_input)

    return "</br>".join(visualized_inputs)
