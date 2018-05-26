from Augmentor.Operations import Operation

# Create your new operation by inheriting from the Operation superclass:
class jitter(Operation):
    def __init__(self, probability, num_of_folds):
        Operation.__init__(self, probability)


    # Your class must implement the perform_operation method:
    def perform_operation(self, image):

        return image