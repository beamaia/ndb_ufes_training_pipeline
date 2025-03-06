class EarlyStopper:
    """
    Stops after either no loss improvements since best loss (mode all)
    or after no loss improvements of size delta positive or negative (mode local)
    """
    def __init__(self, patience=20, min_delta=0.001, mode="all"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.mode = mode

    def early_stop(self, validation_loss):
        
        if self.mode == "local":
            diff = abs(validation_loss - self.min_validation_loss)
            if validation_loss < self.min_validation_loss and diff > self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif diff <= self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        else:
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss >= (self.min_validation_loss + self.min_delta):
                
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False