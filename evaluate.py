import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from data_loader import get_default_device, to_device, testloader
from model import ResNet9

def test_label_predictions(model, device, test_loader, rep_size=None):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple) and rep_size is not None:
                output = output[rep_size]  # Use the specified representation size for evaluation
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

def plot_accuracies(sizes, accuracies):
    plt.figure()
    plt.plot(sizes, [accuracies[size] for size in sizes], marker='o')
    plt.xlabel('Representation Size')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Top-1 Accuracy for Different Representation Sizes')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xscale('log', base=2)  # Use a logarithmic scale with base 2 for the x-axis
    plt.xticks(sizes, [str(size) for size in sizes])  # Set x-ticks to be exactly the representation sizes
    plt.savefig("representation_accuracies.pdf")
    plt.show()

def evaluate_model():
    device = get_default_device()
    nesting_list = [1028, 512, 256, 128, 64, 32, 16, 8]
    model = ResNet9(3, 100, nesting_list=nesting_list)
    model = to_device(model, device)

    model_path = 'resnet9_mrl.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate on test set for each representation size
    accuracies = {}
    
    for i, size in enumerate(nesting_list):
        current_time = time.time()
        y_test, y_pred = test_label_predictions(model, device, testloader, rep_size=i)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[size] = accuracy
        print(f"Representation Size: {size}, Top-1 Accuracy: {accuracy:.4f}, Evaluation time: {time.time() - current_time:.2f} s")

    # Plot accuracies for different representation sizes
    plot_accuracies(nesting_list, accuracies)

if __name__ == '__main__':
    evaluate_model()
