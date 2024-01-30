{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e533a60d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting tensorflow-text\n",
      "  Downloading tensorflow_text-2.8.1-cp39-cp39-win_amd64.whl (2.5 MB)\n",
      "Requirement already satisfied: tensorflow-hub>=0.8.0 in c:\\notebook\\lib\\site-packages (from tensorflow-text) (0.12.0)\n",
      "Requirement already satisfied: tensorflow<2.9,>=2.8.0 in c:\\notebook\\lib\\site-packages (from tensorflow-text) (2.8.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.1.0)\n",
      "Requirement already satisfied: flatbuffers>=1.12 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (2.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (0.2.0)\n",
      "Requirement already satisfied: h5py>=2.9.0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (3.2.1)\n",
      "Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.1.2)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (3.3.0)\n",
      "Requirement already satisfied: keras<2.9,>=2.8.0rc0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (2.8.0)\n",
      "Requirement already satisfied: setuptools in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (58.0.4)\n",
      "Requirement already satisfied: gast>=0.2.1 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (0.5.3)\n",
      "Requirement already satisfied: absl-py>=0.4.0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.0.0)\n",
      "Requirement already satisfied: libclang>=9.0.1 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (13.0.0)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (0.24.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.45.0)\n",
      "Requirement already satisfied: tensorboard<2.9,>=2.8 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (2.8.0)\n",
      "Requirement already satisfied: tf-estimator-nightly==2.8.0.dev2021122109 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (2.8.0.dev2021122109)\n",
      "Requirement already satisfied: protobuf>=3.9.2 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (3.19.4)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.12.1)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (3.10.0.2)\n",
      "Requirement already satisfied: numpy>=1.20 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.20.3)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.6.3)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\notebook\\lib\\site-packages (from tensorflow<2.9,>=2.8.0->tensorflow-text) (1.16.0)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\notebook\\lib\\site-packages (from astunparse>=1.6.0->tensorflow<2.9,>=2.8.0->tensorflow-text) (0.37.0)\n",
      "Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (0.4.6)\n",
      "Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (1.8.1)\n",
      "Requirement already satisfied: google-auth<3,>=1.6.3 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (2.6.2)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (3.3.6)\n",
      "Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (0.6.1)\n",
      "Requirement already satisfied: werkzeug>=0.11.15 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (2.0.2)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\notebook\\lib\\site-packages (from tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (2.26.0)\n",
      "Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\\notebook\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (0.2.8)\n",
      "Requirement already satisfied: rsa<5,>=3.1.4 in c:\\notebook\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (4.8)\n",
      "Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\\notebook\\lib\\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (5.0.0)\n",
      "Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\\notebook\\lib\\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (1.3.1)\n",
      "Requirement already satisfied: importlib-metadata>=4.4 in c:\\notebook\\lib\\site-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (4.8.1)\n",
      "Requirement already satisfied: zipp>=0.5 in c:\\notebook\\lib\\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (3.6.0)\n",
      "Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\\notebook\\lib\\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (0.4.8)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\notebook\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (1.26.7)\n",
      "Requirement already satisfied: charset-normalizer~=2.0.0 in c:\\notebook\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\notebook\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (3.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\notebook\\lib\\site-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (2021.10.8)\n",
      "Requirement already satisfied: oauthlib>=3.0.0 in c:\\notebook\\lib\\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<2.9,>=2.8.0->tensorflow-text) (3.2.0)\n",
      "Installing collected packages: tensorflow-text\n",
      "Successfully installed tensorflow-text-2.8.1\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import tensorflow_hub as hub\n",
    "!pip install tensorflow-text\n",
    "import tensorflow_text as text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a9bdb89c",
   "metadata": {},
   "outputs": [],
   "source": [
    "bert_preprocess = hub.KerasLayer(\"https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3\")\n",
    "bert_encoder = hub.KerasLayer(\"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "71e6b235",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "be25d576",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(r\"D:\\E\\Baikka\\Project\\KEC SAC radiology data for CS 8.3.2022.csv\",encoding='utf-8')\n",
    "df['mr_report'] = df['mr_report'].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "94172d01",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Outcome</th>\n",
       "      <th>mr_report</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Reason for Exam: CHRONIC LOWER BACK PAIN.  GET...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>Reason for Exam: KNOWN MULTILEVEL DEGENERATIVE...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>MR LUMBAR SPINE    Reason for Exam: PROGRESSIV...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>MR CERVICAL SPINE    Reason for Exam: HAS HX O...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>MRI lumbar spine     Comparison: No prior     ...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Outcome                                          mr_report\n",
       "0        0  Reason for Exam: CHRONIC LOWER BACK PAIN.  GET...\n",
       "1        0  Reason for Exam: KNOWN MULTILEVEL DEGENERATIVE...\n",
       "2        0  MR LUMBAR SPINE    Reason for Exam: PROGRESSIV...\n",
       "3        0  MR CERVICAL SPINE    Reason for Exam: HAS HX O...\n",
       "4        0  MRI lumbar spine     Comparison: No prior     ..."
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0a14e168",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "142    Reason for Exam: R/O IMPINGEMENT SEVERE PAIN R...\n",
       "316    History: Right arm pain, weakness and numbness...\n",
       "212    Reason for Exam: NECK PAIN WITH WORSENING OF R...\n",
       "72     Reason for Exam: 1 MONTH HX OF PAIN RADIATING ...\n",
       "Name: mr_report, dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(df['mr_report'],df['Outcome'], stratify=df['Outcome'])\n",
    "X_train.head(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7f4b2dd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bert layers\n",
    "text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')\n",
    "preprocessed_text = bert_preprocess(text_input)\n",
    "outputs = bert_encoder(preprocessed_text)\n",
    "\n",
    "# Neural network layers\n",
    "l = tf.keras.layers.Dropout(0.1, name=\"dropout\")(outputs['pooled_output'])\n",
    "l = tf.keras.layers.Dense(1, activation='sigmoid', name=\"output\")(l)\n",
    "\n",
    "# Use inputs and outputs to construct a final model\n",
    "model = tf.keras.Model(inputs=[text_input], outputs = [l])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b4d6a4f2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "9/9 [==============================] - 53s 5s/step - loss: 0.4829 - accuracy: 0.8273\n",
      "Epoch 2/2\n",
      "9/9 [==============================] - 44s 5s/step - loss: 0.4595 - accuracy: 0.8417\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x2af9f060220>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
    "model.fit(X_train, y_train, epochs=2, batch_size = 32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "af5c40bd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.12902561 0.10799339 0.1489943  0.10958728 0.12785512 0.15359366\n",
      " 0.12236315 0.10889408 0.10667577 0.10887396 0.10884365 0.13631803\n",
      " 0.11957696 0.09176418 0.10935745 0.11703837 0.12399232 0.14666778\n",
      " 0.17020515 0.11119166 0.10105175 0.11489993 0.12303671 0.11900082\n",
      " 0.11628246 0.1098333  0.11355963 0.11089167 0.12153462 0.11087748\n",
      " 0.1253325  0.15376154 0.14513978 0.11113068 0.1025767  0.12653339\n",
      " 0.13307104 0.09547913 0.13412371 0.11629295 0.12678683 0.1350261\n",
      " 0.12250009 0.13328579 0.12082794 0.14918372 0.14072093 0.12044653\n",
      " 0.13927028 0.115659   0.10242283 0.10629401 0.10756418 0.1713723\n",
      " 0.11397156 0.13143313 0.12987432 0.12911898 0.10530183 0.13270181\n",
      " 0.1727942  0.17298016 0.10355785 0.11688769 0.12946466 0.107728\n",
      " 0.15501672 0.1316188  0.11830351 0.0968731  0.09885541 0.13158682\n",
      " 0.12731162 0.11469534 0.14062062 0.1436153  0.13213447 0.11935875\n",
      " 0.16083208 0.12711576 0.12501052 0.12373018 0.13858604 0.12836272\n",
      " 0.1329805  0.09516835 0.13621691 0.10504806 0.12741715 0.12450475\n",
      " 0.13979164 0.10645407 0.11674413]\n"
     ]
    }
   ],
   "source": [
    "y_predicted = model.predict(X_test)\n",
    "y_predicted = y_predicted.flatten()\n",
    "print(y_predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "21dc5e8a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
