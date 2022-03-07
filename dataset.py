from mpose import MPOSE
import numpy as np

def generate_dataset():
    """
    Create MPOSE2021 dataset object and save it into .npy files
    """
    dataset = MPOSE(pose_extractor='openpose', 
                    split=1, 
                    preprocess='scale_and_center', 
                    remove_zip=False)
                    

    # Reduce pose keypoints
    dataset.reset_data()
    dataset.reduce_keypoints()
    dataset.scale_and_center()
    dataset.remove_confidence()
    dataset.flatten_features()

    # Save samples as numpy arrays
    X_train, y_train, X_test, y_test = dataset.get_data()
    # print(X_train.shape, X_test.shape)

    np.save('data/X_train', X_train )
    np.save('data/y_train',y_train)
    np.save('data/X_test',X_test)
    np.save('data/y_test',y_test)


if __name__ == "__main__":
    print("getting MPOSE2021 dataset ...")
    generate_dataset()
    print("finished !")