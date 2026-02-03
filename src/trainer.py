import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, iou_metric):
    model.train()
    total_loss = 0
    iou_metric.reset()

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device).contiguous()
        masks = masks.to(device).long().contiguous()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        iou_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute()

def validate(model, loader, criterion, device, iou_metric, f1_metric):
    model.eval()
    total_loss = 0
    iou_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device).contiguous()
            masks = masks.to(device).long().contiguous()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)
            f1_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute(), f1_metric.compute()import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, iou_metric):
    model.train()
    total_loss = 0
    iou_metric.reset()

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device).contiguous()
        masks = masks.to(device).long().contiguous()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        iou_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute()

def validate(model, loader, criterion, device, iou_metric, f1_metric):
    model.eval()
    total_loss = 0
    iou_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device).contiguous()
            masks = masks.to(device).long().contiguous()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)
            f1_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute(), f1_metric.compute()import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, iou_metric):
    model.train()
    total_loss = 0
    iou_metric.reset()

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device).contiguous()
        masks = masks.to(device).long().contiguous()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        iou_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute()

def validate(model, loader, criterion, device, iou_metric, f1_metric):
    model.eval()
    total_loss = 0
    iou_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device).contiguous()
            masks = masks.to(device).long().contiguous()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)
            f1_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute(), f1_metric.compute()import torch
import torch.nn as nn
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, criterion, device, iou_metric):
    model.train()
    total_loss = 0
    iou_metric.reset()

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device).contiguous()
        masks = masks.to(device).long().contiguous()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        iou_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute()

def validate(model, loader, criterion, device, iou_metric, f1_metric):
    model.eval()
    total_loss = 0
    iou_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device).contiguous()
            masks = masks.to(device).long().contiguous()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)
            f1_metric.update(preds, masks)

    return total_loss / len(loader), iou_metric.compute(), f1_metric.compute()