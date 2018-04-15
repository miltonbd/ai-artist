from classification.base_classifier import BaseClassifier

class MyCifar10Classifier(BaseClassifier):
    def __init__(self,params):
        BaseClassifier.__init__(self,params)



if __name__ == '__main__':
    params={'batch_size_train_per_gpu':5}
    classifier=MyCifar10Classifier(params)
    classifier.train()
