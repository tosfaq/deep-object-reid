import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Script to check multilabel annotation')
    parser.add_argument('data_path', help='path to the dataset root')
    parser.add_argument('--train_annotation',
                        help='path to the pre-trained model weights',
                        default='train.json')
    parser.add_argument('--val_annotation',
                        help='path to the pre-trained model weights',
                        default='val.json')
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()

    all_classes = []
    root = args.data_path
    subsets=[args.val_annotation, args.train_annotation]

    for subset in subsets:
        with open(os.path.join(root, subset)) as f:
            anno = json.load(f)
            all_classes.append(anno['classes'])
            assert len(set(anno['classes'])) == len(anno['classes'])
            print(f'Source {subset} len: {len(anno["images"])}')

            if args.clean:
                clean_anno = []
                for img_info in anno['images']:
                    if os.path.isfile(os.path.join(root, img_info[0])):
                        clean_anno.append(img_info)
                    else:
                        print(f'Missing image: {img_info}')

                if len(anno['images']) != len(clean_anno):
                    anno['images'] = clean_anno
                    with open(os.path.join(root, subset.split('.')[-2] + '_clean.json', 'w')) as of:
                        json.dump(anno, of)

        if 'train' in subset:
            total_unique_labels = 0
            for record in anno['images']:
                labels = set(record[-1])
                total_unique_labels += len(labels)

            n_classes = len(anno['classes'])
            n_train_images = len(anno['images'])
            avg_labels = total_unique_labels / n_train_images

            print(f'Train images {n_train_images}\n'
                f'Classes {n_classes}\n'
                f'Avg labels per train image {avg_labels:.2f}')

    for i, clssA in enumerate(all_classes):
        for j, clssB in enumerate(all_classes):
            if j > i:
                assert set(clssA) == set(clssB)


if __name__ == '__main__':
    main()
