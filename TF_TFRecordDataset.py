import tensorflow as tf
import pandas as pd
import csv

# cols_strings='tarif_type,counter_number,counter_statue,counter_code,reading_remarque,counter_coefficient,consommation_level_1,consommation_level_2,consommation_level_3,consommation_level_4,old_index,new_index,months_number,counter_type'
# cols_names= cols_strings.split(',')
df= pd.read_csv('train.csv',header=0
                    # ,iterator=True,usecols=cols_names
                    )
def cvs_generator(file_name):
    with open(file_name,'r') as f:
        reader= csv.reader(f)
        next(reader)
        for row in reader:
            tmp= row.split(',')
            yield tmp

"""
tf.train.Example is the proto buff massage with single input known as features
tf.train.Example(features=tf.train.Features(dictionary of single or multiple features))
tf.train.Features is the proto buff massage and single input to proto buff example
tf.train.feature can be a single feature or list of features ==> tf.train.FloatList or ByteList or IntList 
"""
def get_cat_numeric_index():
    cols = df.dtypes.tolist()
    cols= cols[0:-1]
    cat_index = []
    num_index= []
    for i, col in enumerate(cols):
        if col=='object':
            cat_index.append(i)
        else:
            num_index.append(i)
    return cat_index,num_index

cat_index,num_index= get_cat_numeric_index()
len_cat_index=len(cat_index)
len_num_index= len(num_index)
def create_tf_records(row):
    numeric_features= row[num_index]
    category_deatures= row[cat_index]
    "Name of arguments must be exactly same as below or better to say "
    "Name of arguments must follow proto buff naming"
    features_dict={
        'numeric_features' : tf.train.Feature(float_list=tf.train.FloatList(value=numeric_features)),
        'categorical_features': tf.train.Feature(bytes_list=tf.train.BytesList(value=list(map(lambda x: x.encode('utf8') if type(x)==str else ''.encode('utf8'),category_deatures )) )),
        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[row[-1]]))
    }
    proto_buff_example= tf.train.Example(features=tf.train.Features(feature=features_dict))
    return proto_buff_example.SerializeToString()

def create_tf_dataset(df):
    with tf.io.TFRecordWriter('tfdata.tfrecords') as writer:
        for row in df.values:
            serialized_rec= create_tf_records(row)
            writer.write(serialized_rec)

def parse_tfrecords(example_proto_buf):
    map_dict={
        'numeric_features': tf.io.FixedLenFeature((len_num_index,),tf.float32),
        'categorical_features' : tf.io.FixedLenFeature((len_cat_index,),tf.string),
        'label': tf.io.FixedLenFeature((),tf.int64)
    }
    parsed_features= tf.io.parse_single_example(example_proto_buf,map_dict)
    return parsed_features['numeric_features'],parsed_features['categorical_features'], \
           parsed_features['label']

dataset= tf.data.TFRecordDataset(['tfdata.tfrecords'])
dataset=dataset.map(parse_tfrecords)

