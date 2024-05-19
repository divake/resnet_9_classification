import time
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
from train import model, device, testloader, trainloader, evaluate

def evaluate_model():
    model_path = 'group22_pretrained_model.h5'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate on test set
    current_time = time.time()
    result = evaluate(model, testloader)
    print("Evaluation time: {:.2f} s".format(time.time() - current_time))
    
    y_test, y_pred = test_label_predictions(model, device, testloader)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    fs = f1_score(y_test, y_pred, average='weighted')
    rs = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print('Confusion matrix:')
    print(cm)
    print(classification_report(y_test, y_pred))
    print('F1 score: %f' % fs)
    print('Recall score: %f' % rs)
    print('Accuracy score: %f' % accuracy)

    # Save classification report into CSV
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('classification_report.csv', index=False)

    # Plot and save classification report
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    plot_classification(precision, recall, f1)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm)

    # Evaluate on train set for training accuracy
    y_train, y_pred_train = test_label_predictions(model, device, trainloader)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print('Train accuracy: %f' % train_accuracy)

def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

def plot_classification(precision, recall, f1_score):
    plt.rcParams['font.size'] = 12
    plt.rc('axes', linewidth=1.75)
    marker_size = 8
    figsize = 6
    plt.figure(figsize=(1.4 * figsize, figsize))
    plt.subplot(3, 1, 1)
    plt.plot(precision, 'o', markersize=marker_size)
    plt.ylabel('Precision', fontsize=14)
    plt.xticks([])
    plt.subplot(3, 1, 2)
    plt.plot(recall, 'o', markersize=marker_size)
    plt.ylabel('Recall', fontsize=14)
    plt.xticks([])
    plt.subplot(3, 1, 3)
    plt.plot(f1_score, 'o', markersize=marker_size)
    plt.ylabel('F1-score', fontsize=14)
    plt.xlabel('Class', fontsize=14)
    plt.subplots_adjust(hspace=0.001)
    plt.tight_layout()
    plt.savefig("classification.pdf")

def plot_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.savefig("confusion_matrix.pdf")
    plt.show()

if __name__ == '__main__':
    evaluate_model()
