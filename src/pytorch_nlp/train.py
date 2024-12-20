import csv

import torch


def save_metrics(train_losses, val_losses, val_accuracies, metrics_path):
    # Save metrics to a CSV file
    with open(metrics_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])
        for epoch, (train_loss, val_loss, val_accuracy) in enumerate(
            zip(train_losses, val_losses, val_accuracies), 1
        ):
            writer.writerow([epoch, train_loss, val_loss, val_accuracy])

    print(f"Metrics saved to {metrics_path}")


def load_model(model, model_path, device):
    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=device)
    )
    print(f"Model loaded from {model_path}")

    return model


def load_metrics(metrics_path):
    # Load metrics from a CSV file
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []

    with open(metrics_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            epochs.append(int(row[0]))  # Epoch
            train_losses.append(float(row[1]))  # Train Loss
            val_losses.append(float(row[2]))  # Validation Loss
            val_accuracies.append(float(row[3]))  # Validation Accuracy

    print(f"Metrics loaded from {metrics_path}")

    # Return the metrics as lists
    return epochs, train_losses, val_losses, val_accuracies


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Perform one epoch of training."""
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())

        # Backward pass
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    return avg_train_loss


def evaluate_model(model, val_loader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            total_val_loss += loss.item()

            # Compute accuracy
            predictions = (outputs.squeeze() >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct / total
    return avg_val_loss, val_accuracy


def save_best_model(
    model, val_accuracy, best_val_accuracy, patience_counter, model_path
):
    """Check if the current validation loss is the best and save the model."""
    if val_accuracy > best_val_accuracy:
        torch.save(model.state_dict(), model_path)
        print("  Validation accuracy improved. Model saved!")
        return val_accuracy, 0  # Reset patience counter
    else:
        patience_counter += 1
        print(
            f"  No improvement in validation accuracy. Patience counter: {patience_counter}"
        )
        return best_val_accuracy, patience_counter


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    patience,
    early_stopping,
    output,
    run_name,
):
    """Train the model with early stopping and track training history."""
    model.to(device)
    best_val_accuracy = float(0)
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    model_path = f"{output}/{run_name}.pt"
    metrics_path = f"{output}/{run_name}.csv"

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.4f}")

        # Check for early stopping
        best_val_accuracy, patience_counter = save_best_model(
            model, val_accuracy, best_val_accuracy, patience_counter, model_path
        )
        if early_stopping and patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Load the best model after training
    model = load_model(model, model_path, device)
    print("Best model loaded!")

    save_metrics(train_losses, val_losses, val_accuracies, metrics_path)

    # Return training history for visualization
    return train_losses, val_losses, val_accuracies
