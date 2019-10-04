# add_faces_to_svm_class
# Incremental training for Class svm With [facenet](https://github.com/davidsandberg/facenet)

Sometimes we need to build a model of svm or any other type,
Our data size is often very large
Every time we need to add a new group of persons
You will have to start train to all the data again

This is impossible

Because training time takes a very long time


# Why here?

# First,check out this python [notebook](https://github.com/mohammed-Emad/add_faces_to_svm_class/blob/master/add_clasifir_class.ipynb)and try everything

The reason is that time extraction features across neural network models (CNN)
It takes too long

In a project like facenet the form is eventually filled in to the class svm
This is for the purpose of using one of the best tools to learn machine in the classification of faces later

The form is saved in the end to a file type *.pkl (pickle)

But what is saved?

Features extracted from images are saved after arranging them based on their location within the data labels and class names 

```python
   with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
```

It can be done through one simple write at the end of the file or append data to file with the pickle type
This is done by adding 'ab' to the file format function to write instead of "wb"
So for example

```python
   with open(classifier_filename_exp, 'ab') as outfile:
                    pickle.dump((model, class_names), outfile)
```

Look here the picture is clearer
[append to pickle](https://stackoverflow.com/questions/28077573/python-appending-to-a-pickled-list?rq=1)


For class names it is normal to add but for the model
Here's the problem
I tried to search about the possibility of adding elements to the class SVM
But that did not work very well

Another very important thing to talk about is the distribution of data at each training stage how it looks

This is of course, regardless of whether there is a possibility  add new data to svm model

In order to clarify things we will do a simple simulation of what the code does in (facenet)

We will now talk about the training and classification file, which is located here [classifier.py](https://github.com/davidsandberg/facenet/blob/master/src/classifier.py)

First, we use data division to obtain accuracy or not
The data must pass through this line
```
dataset = facenet.get_dataset (args.data_dir)
```

Getting data here is a bit different than usual
A class is created for each person by name. In contrast, the names and paths of the images in each person's folder are obtained
The names of variables in DataSet is:

```
ImageClass(name, image_paths)
```
```
for cls in dataset:
   print("name:",cls.name)
   print("paths:",cls.image_paths)
```

```
#example
name :  Hossam
paths :  ['data_algin/Hossam/Hossam-0.jpg']
name :  biden
paths :  ['data_algin/biden/biden-1.jpg', 'data_algin/biden/biden-0.jpg']
name :  rose_leslie
paths :  ['data_algin/rose_leslie/rose_leslie-1.jpg', 'data_algin/rose_leslie/rose_leslie-4.jpg']
name :  kit_harington
paths :  ['data_algin/kit_harington/kit_harington-0.jpg', 'data_algin/kit_harington/kit_harington-1.jpg']
```

Here seem to have several photos for each of them
We must fill each image according to the person in the picture

Here another function was used to do this
See below

```
paths, labels = facenet.get_image_paths_and_labels(dataset)
```

The previous line gets our data complete and returns two variables to us
One for the tracks and the other for the labels

But what are the labels
The labels are filled in to the place of the person's name as arranged in the previous step inside "ImageClass"

How? Looks like "labels"؟

```
print("paths",paths)
print("labels",labels)
```
```
paths ['data_algin/Hossam/Hossam-0.jpg', 'data_algin/biden/biden-0.jpg', 'data_algin/biden/biden-1.jpg', 'data_algin/rose_leslie/rose_leslie-0.jpg', 'data_algin/rose_leslie/rose_leslie-1.jpg', 'data_algin/rose_leslie/rose_leslie-2.jpg', 'data_algin/rose_leslie/rose_leslie-3.jpg', 'data_algin/rose_leslie/rose_leslie-4.jpg', 'data_algin/kit_harington/kit_harington-0.jpg', 'data_algin/kit_harington/kit_harington-1.jpg']

labels [0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4]

```

Here is where each number appears to indicate the order of the person's name
This is according to all the pictures
see
```
   class_name = ['Hossam','biden' ,'rose_leslie' ,'kit_harington']
   labels =[
   Hossam-0.jpg = 0
   biden-0.jpg  = 1
   biden-1.jpg  = 1
   rose_leslie-0.jpg = 2
   rose_leslie-1.jpg = 2
   rose_leslie-2.jpg = 2
   rose_leslie-3.jpg = 2
   rose_leslie-4.jpg = 2
   kit_harington-0.jpg = 3
   kit_harington-0.jpg = 3]
   
   
```

Another way you can see how most data looks
```
print("class_names" ,len(class_names))
print("labels" ,len(labels))
print("emb_array" ,len(emb_array))
print("\n")
```
class_names 19
labels 665
emb_array 665
```
print("class_names" ,class_names)
print("labels" ,labels)
print("emb_array" ,emb_array)
```

class_names 
['Ariel Sharon', 'Arnold Schwarzenegger', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush', 'Gerhard Schroeder', 'Gloria Macapagal Arroyo', 'Hugo Chavez', 'Jacques Chirac', 'Jean Chretien', 'Jennifer Capriati', 'John Ashcroft', 'Junichiro Koizumi', 'Laura Bush', 'Lleyton Hewitt', 'Luiz Inacio Lula da Silva', 'Serena Williams', 'Tony Blair', 'Vladimir Putin']

labels 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]

emb_array 
[[ 0.07976267  0.09515638  0.05864169 ... -0.05250152 -0.04106444
  -0.04610822]
 [ 0.06595863  0.07263391  0.02989905 ... -0.07689951 -0.06294088
  -0.02233639]
 [ 0.07013855  0.1277086   0.06934369 ... -0.06474113 -0.01231081
   0.01420595]
 ...
 [ 0.11499634 -0.03017467  0.00925456 ... -0.01983628 -0.10197467
  -0.03755766]
 [ 0.03864072 -0.08472022 -0.13453087 ... -0.01985509 -0.04908967
  -0.02526135]
 [ 0.09709937 -0.04564065 -0.02634468 ...  0.00079466 -0.07019936
  -0.07296239]]
  
# done 


Here the features have become somewhat clear

We are talking about a small problem
When new training is conducted, new people get a new arrangement
The order of the number will be 0
That is why he will replace old people
Because they started their order of number 0 too

Eg someone like :Arnold Schwarzenegger
He got the number :1
In a table :labels


A new person came and arranged the same order :1 name "Blair"
So here the discrepancy will occur in the performance where he will identify people and introduce names other than those people

Here the solution was to save the features in their original form unchanged

Means saving photo features
Save their order in the table
Save contacts

When we want to start incremental training
We will upload the file "all_features.pkl" and write from where we finished in the past

Here was the use of four or five lines to do this
See their place here [my_clasifiy.py](https://github.com/mohammed-Emad/add_faces_to_svm_class/blob/master/my_clasifiy.py#L191)

Here is the first line skipping the last number in the table :labels

```
labels_n = np.array(labels_n) + max(labels_old) +1 

#example 
labels_old = [0,0,1,1,2,2,3,3,3,3,4,4,5,5]
labels_n = [0,0,0,1,1,1,2,2,2,3,3,4,5]
max(labels_old) =5        # Error count 5!!<-^
labels_n = np.array(labels_n) +5
labels_n = [5,5,5,6,6,6,7,7,7,8,8,9,10] #cunt 5?
   #see [0,0,1,1,2,2,3,3,3,3,4,4,5,5] + [5,5,5,6,6,6,7,7,7,8,8,9,10] #error
labels_n = np.array(labels_n) +5 +1 #add 1
labels_n = [6,6,6,7,7,7,8,8,8,9,9,10,11]
labels_n = np.concatenate((labels_old ,labels_n.tolist())).tolist() #labels_old + labels_n

labels_n = [0,0,1,1,2,2,3,3,3,3,4,4,5,5,6,6,6,7,7,7,8,8,8,9,9,10,11]

```
The next is adding people's names
And features extracted from images

```
          class_n = np.concatenate((class_old ,class_n))        #append new class name to old class names
          emb_n = np.concatenate((emb_old ,emb_n))              #add new labels to old labels
          
```
The features are then saved to the file again
Then the names of the characters and the features of all the old and new images are rearranged and the images are arranged in the table to be converted to the svm class

```
if os.path.isfile(args.features_filename) and (args.training_type == 'incremental'): # -----<--new
          stack_old_data ,dataset = Remove_duplicate_names(args ,dataset) #----<--new
```

There is something else
Be sure not to repeat the same person twice in a gradual training
The following function was created to do this 
```
About 7 files of data were deleted for about 2 people whose names already existd
or
No one has been deleted(0)
```
They delete people whose names have been repeated again in a new training with the notice that several people have been deleted or not


All this is done in two functions that were created inside the file mentioned earlier [my_clasifiy.py](https://github.com/mohammed-Emad/add_faces_to_svm_class/blob/master/my_clasifiy.py#L191)

The initial function is
```
def Remove_duplicate_names(args ,dataset):
    image_del = 0
    name_deleted = []
    stak_host = []
    dataset_n =[]
    features_filename = os.path.expanduser(args.features_filename)

    print("load_features from file :'%s'" % args.features_filename)
    with open(features_filename, 'rb') as infile:
             (emb_old ,labels_old ,class_old) = pickle.load(infile)

    if len(class_old) > 0:
      for cls in dataset:
          name = cls.name.replace('_', ' ')    #get the class name
          if name in class_old:                #find
             name_deleted.append(name) 
             image_del +=len(cls.image_paths)

          else:
             dataset_n.append(facenet.ImageClass(cls.name, cls.image_paths))

    if image_del > 0:

       #Remove duplicate names in the deletion list
       ResultList = sorted(set(name_deleted), key=lambda x:name_deleted.index(x))
       print("About (%s) files of data were deleted for about '%s' people whose names already existd"% image_del ,len(ResultList))

    else:
       print("No one has been deleted(%s)" % len(name_deleted))
    stak_host = emb_old ,labels_old ,class_old
    return  stak_host ,dataset
```
The other function is

```
def Incremental_training(args ,stack_old_data ,emb_n ,labels_n ,class_n): #-----<-- new
    
    features_filename = os.path.expanduser(args.features_filename)

    
    if os.path.isfile(args.features_filename) :

       emb_old ,labels_old ,class_old = stack_old_data
       if len(class_old) >0:
          print("start Incremental_training")
          labels_n = np.array(labels_n) + max(labels_old) +1                      # move numbers new list to old list end
          labels_n = np.concatenate((labels_old ,labels_n.tolist())).tolist()     #add new labels to old labels
          class_n = np.concatenate((class_old ,class_n))        #append new class name to old class names
          emb_n = np.concatenate((emb_old ,emb_n))              #append new image feature to old images feature
       
    else:
        print("not found file!:%s and start not Incremental_training" %args.features_filename)

    with open(features_filename, 'wb') as outfile:
         pickle.dump((emb_n ,labels_n ,class_n), outfile, pickle.HIGHEST_PROTOCOL)

    print('Saved a features as model to file "%s"' % args.features_filename)
    return emb_n ,labels_n ,class_n
```
As for run, there is another addition to the input commands with the code
see
```#the lines in fun "parse_arguments()"
        parser.add_argument('--training_type', type=str, choices=['incremental', 'normal'], #------------<--new1
        help='Indicates whether you want to gradually train your data Or normal training.' + 
        'In the extra training you will need to provide the location of the features file.' +
        'informing you that it will be overwritten every time.', default='normal')
        #and
        parser.add_argument('--features_filename', # --------------<<----new2
        help='Features extracted file name as a pickle (.pkl) file. ' + 
        'path and name to file Features extracted and classes names and lable.', default='none')
```
# how to use!
# ----------------------------------
Please note that the update on the run orders is only two
```
--training_type incremental 
--features_filename ~/my_model/all_features.pkl 

```

For this when running, use the following command

```
!python my_clasifiy.py \
TRAIN  \
--training_type incremental \
~/datasets/lfw/lfw_mtcnnpy_160 \
~/datasets/20180402-114759/20180402-114759.pb  \
~/my_model/lfw_classifier.pkl \
--features_filename ~/my_model/all_features.pkl \
--batch_size 1000 \
--min_nrof_images_per_class 40 \
--nrof_train_images_per_class 35 \
--use_split_dataset
```

Check the following options. You probably will not need to type them

```
--min_nrof_images_per_class 40 \
--nrof_train_images_per_class 35 \
--use_split_dataset
```
**Once again remind you to see the notebook [add_clasifir_class.ipynb](https://github.com/mohammed-Emad/add_faces_to_svm_class/blob/master/add_clasifir_class.ipynb)**

Latest Disclaimer
This is not the only way, there are more other ways
But the idea here is to save features after training on a set of images
You do not need to train a million images each time you want to add a new group of people

You can change whatever you want
But remember the whole thing is to save the extracted features to write on it again

Because the longest time is wasted is the extraction time of the features of the images
So keep it in a safe place {emb_array!!}
```
The next step is the extra training of the faces model resnet? [exsample 20180402-114759.pd]
```
This idea has been before by M & S
This may be useful to someone
Thanks all
