import tensorflow as tf
import pandas as pd

# cols_strings='tarif_type,counter_number,counter_statue,counter_code,reading_remarque,counter_coefficient,consommation_level_1,consommation_level_2,consommation_level_3,consommation_level_4,old_index,new_index,months_number,counter_type'
# cols_names= cols_strings.split(',')
df= pd.read_csv('train.csv',header=0
                    # ,iterator=True,usecols=cols_names
                    )

# df= reader.get_chunk(10000)
def cat_to_num(cat_list):
    out=[]
    for col in cat_list:
        lst=[i for i in df[col].unique().tolist() if type(i)==str]
        tmp=dict([(val.encode('utf8'), i) for i, val in enumerate(lst)])
        out.append(tmp)
    return out

def get_defaults():
    cols= df.columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).\
        columns.tolist()
    stats = df.describe()
    default_num = dict([(col, stats.loc['50%', col]) for col in num_cols])
    defult_cat = dict([(col, df[col].value_counts().sort_values(ascending=False).index[0]) for col in cat_cols])
    default_num.update(defult_cat)
    default_values=[]
    for col in cols:
        default_values.append(default_num[col])
    return default_values

def get_cat_index():
    cols = df.dtypes.tolist()
    cat_index = []
    for i, col in enumerate(cols):
        if col=='object':
            cat_index.append(i)
    return cat_index

def create_tf_dictionary(list_dicts):
    tf_list_dicts=[]
    for dic in list_dicts:
        tf_list_dicts.append(tf.lookup.StaticHashTable(initializer=
        tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(dic.keys())),
        values=tf.constant((list(dic.values())), dtype=tf.float32)),
        default_value=tf.constant(0, dtype=tf.float32)))
    return tf_list_dicts


def parse_csv(row):
    r=tf.io.decode_csv(row,def_)
    r=r[2:]
    for i,idx in enumerate(cat_index):
        r[idx]= tf_dicts[i].lookup(r[idx])
    x = tf.stack(r[0:-1], -1)
    y = tf.stack([r[-1]], -1)
    return x,y


def_= [[i] for i in get_defaults()]
cat_list=cat_to_num(df.select_dtypes(include=['object']).columns.tolist())
cat_index= get_cat_index()
tf_dicts= create_tf_dictionary(cat_list)
Batch_Size= 2
dataset = tf.data.TextLineDataset('train.csv').skip(1).map(parse_csv)
dataset= dataset.batch(Batch_Size)