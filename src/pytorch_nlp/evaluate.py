import torch


def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.0  # Track total loss for averaging

    with torch.no_grad():
        for batch in test_loader:
            # Unpack the batch and move to the correct device
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(input_ids)
            loss = criterion(outputs.squeeze(), labels)  # Calculate the loss

            total_loss += loss.item()  # Accumulate loss for averaging

            # Predictions using sigmoid threshold at 0.5 for binary classification
            predictions = (outputs.squeeze() >= 0.5).float()

            # Update accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)  # Calculate the average loss

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")  # Print average loss
    print(f"Correctly classified samples: {correct}")
    print(f"Wrongly classified samples: {total - correct}")

    return accuracy, avg_loss
