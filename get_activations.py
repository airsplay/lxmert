import os
import collections

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from src.get_activations_model import NLVR2Model
from src.get_activations_data import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = NLVR2Dataset(splits)
    tset = NLVR2TorchDataset(dset)
    evaluator = NLVR2Evaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

class Activations:
    def __init__(self):
        self.name = "World"
        args.load_lxmert = r'C:\Users\asmit\Desktop\Guided Research\lxmert\model\model'
        self.model = NLVR2Model()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats, boxes
                logit = self.model(feats, boxes, sent)
                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l
        #if dump is not None:
            #evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


    def print_message(self):
        print("Args.load: ",args.load_lxmert)


if __name__ == '__main__':
    '''
    act = Activations()
    act.print_message()'''
    #print("Args.load: ",args.load_lxmert)
    args.load_lxmert = 'load_lxmert'
    activations = Activations()

    #Give the path to the pre-trained model : model\model_LXRT
    if args.load is not None:
        activations.load(args.load)

    #Test

    result = activations.evaluate(
        get_tuple(args.test, bs=args.batch_size,
                  shuffle=False, drop_last=False),
        #dump=os.path.join(args.output, '%s_predict.csv' % args.test)
    )
    print(result)


