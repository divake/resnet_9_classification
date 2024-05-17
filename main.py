import train
import evaluate

def main():
    train.train_model()
    evaluate.evaluate_model()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
