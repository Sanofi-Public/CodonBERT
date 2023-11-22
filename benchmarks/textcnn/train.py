import torch
import torch.autograd as autograd
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def train(train_iter, dev_iter, test_iter, model, args):
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_loss = 100000
    best_test_loss = 100000

    best_dev_acc = -100000
    best_test_acc = -100000

    best_epoch = -1
    best_model = None
    for epoch in tqdm(range(1, args.epochs + 1)):
        for batch in train_iter:
            model.train()
            feature, target = batch
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            if args.labels > 1:  # classification
                target = torch.squeeze(target)
                loss = F.cross_entropy(logit, target)
            else:  # regression
                assert args.labels == 1
                loss = F.mse_loss(logit, target)

            loss.backward()
            optimizer.step()

        dev_loss, dev_acc = eval(dev_iter, model, args, "Evaluation")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_acc = dev_acc
            best_epoch = epoch

            # save model if save_path exists
            best_model = model.state_dict()

            # evaluate the test set
            best_test_loss, best_test_acc = eval(test_iter, model, args, "Test")

    if args.labels == 1:
        print(
            "Best epoch: {} | {:s} - MSE loss: {:.6f}, spearman correlation: {:.6f} | {:s} - MSE loss: {:.6f}, spearman correlation: {:.6f}".format(
                best_epoch,
                "Eval set",
                best_dev_loss,
                best_dev_acc,
                "Test set",
                best_test_loss,
                best_test_acc,
            )
        )
    else:
        assert args.labels > 1
        print(
            "Best epoch: {} | {:s} - cross-entropy loss: {:.6f}, accuracy: {:.6f} | {:s} - cross-entropy loss: {:.6f}, accuracy: {:.6f}".format(
                best_epoch,
                "Eval set",
                best_dev_loss,
                best_dev_acc,
                "Test set",
                best_test_loss,
                best_test_acc,
            )
        )

    if args.save_path:
        torch.save(best_model, args.save_path)
        print("Saving the best model to %s." % args.save_path)


def eval(data_iter, model, args, datatype):
    model.eval()
    corrects, sum_loss = 0, 0
    prds = []
    tgts = []
    for batch in data_iter:
        feature, target = batch
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)

        # classification
        if args.labels > 1:
            target = torch.squeeze(target)
            loss = F.cross_entropy(logit, target)
        else:
            assert args.labels == 1
            loss = F.mse_loss(logit, target)

        sum_loss += loss.item()

        if args.labels > 1:
            logit = torch.argmax(logit, dim=1)
        prds.extend(torch.flatten(logit).tolist())
        tgts.extend(torch.flatten(target).tolist())

    # classification
    if args.labels > 1:
        corr = accuracy_score(prds, tgts)
    else:
        assert args.labels == 1
        corr, _ = stats.spearmanr(prds, tgts)

    return sum_loss / len(data_iter), corr


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item() + 1]
