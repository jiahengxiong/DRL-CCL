import torch
from torchviz import make_dot
from model import OneStepRNN

def draw_model_graph():
    model = OneStepRNN(input_dim=1, hidden_dim=64)
    model.eval()

    # fake input: (B=1, T=5, 1)
    x = torch.randn(1, 5, 1)

    # forward
    y = model(x)

    # draw graph
    dot = make_dot(
        y,
        params=dict(model.named_parameters())
    )
    dot.render("onestep_rnn_graph", format="pdf")

if __name__ == "__main__":
    draw_model_graph()