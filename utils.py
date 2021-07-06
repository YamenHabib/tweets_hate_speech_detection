import torch


#object to hold out our results and to save and reload model and metrics file
class ResultsSaver():
  def __init__(self, train_len, val_len,output_path, device):
    self.train_losses = []
    self.val_losses = []
    self.steps = []
    
    self.best_val_loss = float('Inf')
    
    self.train_len = train_len
    self.val_len = val_len
    
    self.output_path = output_path
    self.device = device
  
  def save_checkpoint(self, path, model, valid_loss):
    torch.save({'model_state_dict': model.state_dict(),'valid_loss': valid_loss}, self.output_path + path)

  def load_checkpoint(self, path, model):    
    state_dict = torch.load(self.output_path + path, map_location=self.device)
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

  def save_metrics(self, path):   
    state_dict = {'train_losses': self.train_losses,
                  'val_losses': self.val_losses,
                  'steps': self.steps}
    
    torch.save(state_dict, self.output_path + path)
  
  def load_metrics(self, path):    
    state_dict = torch.load(self.output_path + path, map_location=self.device)
    return state_dict['train_losses'], state_dict['val_losses'], state_dict['steps']

  def update_train_val_loss(self, model, train_loss, val_loss, step, epoch, num_epochs):

    train_loss = train_loss / self.train_len
    val_loss = val_loss / self.val_len
    self.train_losses.append(train_loss)
    self.val_losses.append(val_loss)
    self.steps.append(step)
    

    print('Epoch [{}/{}], step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}' .format(epoch+1, num_epochs, step, num_epochs * self.train_len, train_loss, val_loss))
    
    # checkpoint
    if self.best_val_loss > val_loss:
        self.best_val_loss = val_loss
        self.save_checkpoint('/model.pkl', model, self.best_val_loss)
        self.save_metrics('/metric.pkl')


# defin training procedure
def train(model, optimizer, train_iter, valid_iter, results, pad_index, scheduler = None, num_epochs = 5 , train_whole_model = False):
    step = 0
    # if we want to train all the model (our added layer + roBERTa)
    if train_whole_model:
      for param in model.roberta.parameters():
        param.requires_grad = True
    # in case we just want to train our added layer.
    else:
      for param in model.roberta.parameters():
        param.requires_grad = False
    
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0.0                
        val_loss = 0.0
        for (source, target), _ in train_iter:
            mask = (source != pad_index).type(torch.uint8)
            y_pred = model(input_ids=source, attention_mask=mask)  
            loss = torch.nn.CrossEntropyLoss()(y_pred, target)
            loss.backward()
            # Optimizer and scheduler step
            optimizer.step()    
            scheduler.step()
            optimizer.zero_grad()
            # Update train loss and step
            train_loss += loss.item()
            step += 1

        model.eval()
        with torch.no_grad():                    
            for (source, target), _ in valid_iter:
                mask = (source != pad_index).type(torch.uint8)
                y_pred = model(input_ids=source,  attention_mask=mask)
                loss = torch.nn.CrossEntropyLoss()(y_pred, target)
                val_loss += loss.item()
        results.update_train_val_loss(model, train_loss, val_loss, step, epoch, num_epochs)       
        model.train()

    results.save_metrics('/metric.pkl')


def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (source, target), _ in test_loader:
                mask = (source != PAD_INDEX).type(torch.uint8)
                
                output = model(source, attention_mask=mask)

                y_pred.extend(torch.argmax(output, axis=-1).tolist())
                y_true.extend(target.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()

    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.xaxis.set_ticklabels(['Hate', 'Normal'])
    ax.yaxis.set_ticklabels(['Hate', 'Normal'])
