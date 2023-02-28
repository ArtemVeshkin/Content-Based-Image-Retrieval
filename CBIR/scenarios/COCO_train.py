from pycocotools.coco import COCO


def COCO_train(cfg):
    coco = COCO('/home/artem/data/COCO/annotations/image_info_test2017.json')
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
