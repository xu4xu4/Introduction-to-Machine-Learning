from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from math import exp
import random
from math import pi
import warnings
warnings.filterwarnings("ignore")
NumLayer=1
class Preprocessor:
    def __init__(self, train):
        # Initialize the preprocessor with a DataFrame
        self.train = train
        self.val=None

    def preprocess(self):
        # Apply various preprocessing methods on the DataFrame
        self.train = self._preprocess_numerical(self.train)
        self.train = self._preprocess_categorical(self.train)
        self.train = self._preprocess_ordinal(self.train)
        return self.train

    def _preprocess_numerical(self, df):
        # Custom logic for preprocessing numerical features goes here
        df = self.train.copy()
        numerical_cols = df.iloc[:, 0:17].columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

        # Normalize the numerical features to the range [0, 1]
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df
    def _preprocess_val_numerical(self, df):
        df = self.val.copy()
        # Custom logic for preprocessing numerical features goes here
        numerical_cols = df.iloc[:, 0:17].columns
        df[numerical_cols] = df[numerical_cols].fillna(self.train[numerical_cols].mean())

        # Normalize the numerical features to the range [0, 1]
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def _preprocess_categorical(self, df):
        df = self.train.copy()
        # Add custom logic here for categorical features
        #categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        #for col in categorical_features:
            #df[col] = LabelEncoder().fit_transform(df[col])
        categorical_cols = df.iloc[:, 17:].columns
        for col in categorical_cols:
            most_frequent_value = df[col].mode()[0]
            df[col] = df[col].fillna(most_frequent_value)
        return df
    def _preprocess_val_categorical(self, df):
        df = self.val.copy()
        categorical_cols = df.iloc[:, 17:].columns
        for col in categorical_cols:
            most_frequent_value = self.train[col].mode()[0]
            df[col] = df[col].fillna(most_frequent_value)
        return df
    def transform(self, X_val):
        self.val=X_val
        # Use the parameters obtained from the training set to transform the validation set
        # Apply the same transformations as the training set
        self.val = self._preprocess_val_numerical(self.val)
        self.val = self._preprocess_val_categorical(self.val)
        return self.val

    def _preprocess_ordinal(self, df):
        # Custom logic for preprocessing ordinal features goes here
        return df

# Implementing the classifiers (NaiveBayesClassifier, KNearestNeighbors, MultilayerPerceptron)

# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

# Naive Bayes Classifier
class NaiveBayesClassifier(Classifier):
    def __init__(self):
        # Initialize the classifier
        self.x=None
        self.y=None
        self.class_dict=None
        self.class_prob=None
    def fit(self, X, y):
        self.x=X
        self.y=y
        self.x["outcome"]=[y[i]for i,k in self.x.iterrows()]
        self.xx=self.x
        self.xx=self.xx.drop('outcome', axis=1)
        # Compute class probabilities
        kk=[]
        kk2=[]
        for i, row in self.x.iterrows():
            if row[77]==0:
                row=row.drop('outcome')
                kk.append(row)
            else:
                row=row.drop('outcome')
                kk2.append(row)
        self.class_dict={0:np.array(kk),1:np.array(kk2)}
        jj=0
        gg=0
        for i, row in self.x.iterrows():
            if row[77]==0:
                jj=jj+1
            else:
                gg+=1
            self.class_prob={0:(jj/322),1:(gg/322)}
        summaries = dict()
        for class_value, rows in self.class_dict.items():
            mean=np.mean(rows,axis = 0)
            std=np.std(rows,axis = 0)
            m = mean
            s = std
            summaries[class_value] = np.vstack([m,s])
        self.summaries = summaries
        
    def joint_log_likelihood(self, test):
        log_likelihood = []
        for i in np.unique(self.y):
            num1=np.log(self.class_prob[i])
            num2=2*pi*np.square(self.summaries[i][1]+0.00000001)
            num3=test-self.summaries[i][0]
            num4=np.square(self.summaries[i][1]+0.00000001)
            log_class_prob = num1
            log_class_likelihood = -0.5*np.log(num2)
            log_class_likelihood = log_class_likelihood-0.5*(num3)**2/num4
            log_proba_num=num1+np.sum(log_class_likelihood, axis = 0)
            log_likelihood.append(log_proba_num)
        return np.array(log_likelihood).T
    
    def predict(self,X):
        pre=[]
        for i,x in X.iterrows():
            temp = x.to_numpy()
            log_likelihood = self.joint_log_likelihood(temp)
            pre.append(np.argmax(log_likelihood, axis = 0))
        return pre

    def predict_proba(self, X):
        pro=[]
        pr=[]
        for i,x in X.iterrows():
            temp = x.to_numpy()
            log_likelihood = self.joint_log_likelihood(temp)
            pro.append([(log_likelihood[0]+2000)/2000,(log_likelihood[1]+2000)/2000])
        return np.array(pro)

# K-Nearest Neighbors Classifier
class KNearestNeighbors(Classifier):
    def __init__(self, k=9):
        # Initialize KNN with k neighbors
        self.k = k
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):
        # Store training data and labels for KNN
        self.X_train = X
        self.y_train = y
        pass
    def _euclidean_distance(self, x1, x2):
        # Calculate Euclidean distance between two points
        kk=0
        for i in range(0,len(x1)):
            kk+= ((x1[i]-x2[i])**2)

        return np.sqrt(kk)
    def predict(self, X):#x_val
        # Implement the prediction logic for KNN 
        predictions = []
        for temp1,x in X.iterrows():
            dis=[]
            i=0
            for temp2,tt in self.X_train.iterrows():
                di=self._euclidean_distance(x,tt)
                dis.append((di,self.y_train[temp2]))
                i+=1
            dis = sorted(dis)
            dis = np.array(dis[:self.k])
            labels = dis[:, 1]
            uniq_label, counts = np.unique(labels, return_counts=True)
            pred = uniq_label[counts.argmax()]
            predictions.append(pred)
            # Predict the majority class among the k-nearest neighbors
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Implement probability estimation for KNN
        probabilities = []
        for temp1, x in X.iterrows():
            dis = []
            i = 0
            for temp2, tt in self.X_train.iterrows():
                di = self._euclidean_distance(x, tt)
                dis.append((di, self.y_train[temp2]))
                i += 1
            dis = sorted(dis)
            dis = np.array(dis[:self.k])
            labels = dis[:, 1]
            uniq_label, counts = np.unique(labels, return_counts=True)
            class_probabilities = counts / self.k

        # Create an array with consistent shape
            prob_array = np.zeros(len(self.y_train.unique()))
            prob_array[uniq_label.astype(int)] = class_probabilities
            probabilities.append(prob_array)  # Probabilities based on counts

        return np.array(probabilities)
        pass

# Multilayer Perceptron Classifier
class MultilayerPerceptron(Classifier):
    def __init__(self, input_size=77, hidden_layers_sizes=[3,2], output_size=2):
        # Initialize MLP with given network structure
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size

    def fit(self, X, y, epochs=300, learning_rate=0.03):
        # Implement training logic for MLP including forward and backward propagation
        self.x = X
        self.y = y
        self.network=self.initialize_network(self.input_size, self.hidden_layers_sizes, self.output_size)
        for epoch in range(epochs):
            sum_error=0
            for i, row in X.iterrows():
                outputs=self._forward_propagation(row)
                expected = [0 for _ in range(self.output_size)]
                truth=y[i]
                expected[truth] = 1
                #sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self._backward_propagation(expected,outputs)
                self._update_weights(row, learning_rate)
            #print (sum_error)

    def predict(self, X):
        # Implement prediction logic for MLP
        predictions = []
        for i, row in X.iterrows():
            output = self._forward_propagation(row)
            predictions.append(output.index(max(output)))

        return predictions

    def predict_proba(self, X):
        # Implement probability estimation for MLP
        probabilities = []
        for i, row in X.iterrows():
            output = self._forward_propagation(row)
            probabilities.append(output)
        return np.array(probabilities)

    def _forward_propagation(self, row):
        # Implement forward propagation for MLP
        inputs = row
        for layer in self.network:
            outputs=[]
            new_inputs = []
            for neuron in layer:
                activation = self._activate(neuron['weights'], inputs)
                neuron['output'] = self._transfer(activation)
                outputs.append(self._transfer(activation))
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def _backward_propagation(self, expected, outputs):
        # Implement backward propagation for MLP with cross-entropy loss
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                # Calculate errors for hidden layers
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        weight=neuron['weights'][j]
                        delta=neuron['delta']
                        error += weight * delta
                    errors.append(error)
            else:
                # Calculate errors for the output layer with cross-entropy loss
                for j in range(len(layer)):
                    neuron = layer[j]
                    truth=outputs[j]
                    expect=expected[j]
                    errors.append(truth - expect)

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self._transfer_derivative(neuron['output'])

    def _update_weights(self, row, l_rate):
        # Update network weights with error
        inputs = row
        layer_length=len(self.network)
        for i in range(layer_length):
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                weight=len(inputs)
                for j in range(weight):
                    learning_rate=l_rate
                    data=inputs[j]
                    neuron['weights'][j] -= learning_rate * neuron['delta'] * data
                learning_rate=l_rate
                neuron['weights'][-1] -= learning_rate * neuron['delta']

    def initialize_network(self, input_size, hidden_layers_sizes, output_size):
        # Initialize a network
        random.seed(42)
        network = []
        prev_layer_size = input_size
        for layer_size in hidden_layers_sizes:
            std_ = np.sqrt(2 / (prev_layer_size + layer_size))
            layer = [{'weights': [random.uniform(0,std_) for _ in range(prev_layer_size + 1)]} for _ in range(layer_size)]
            network.append(layer)
            prev_layer_size = layer_size

        # Output layer
        std_=np.sqrt(2 / (prev_layer_size + self.output_size))
        output_layer = [{'weights': [random.uniform(0,std_) for _ in range(prev_layer_size + 1)]} for _ in range(output_size)]
        network.append(output_layer)
        return network

    def _activate(self, weights, inputs):
        # Calculate neuron activation
        activation = weights[-1]
        for i in range(len(weights) - 1):
            weight=weights[i]
            row= inputs[i]
            activation += weight * row
        return activation

    def _transfer(self, activation):
        # Transfer neuron activation
        temp=1.0 / (1.0 + exp(-activation))
        return temp

    def _transfer_derivative(self, output):
        # Calculate the derivative of a neuron output
        temp=output * (1.0 - output)
        return temp
# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test)
    print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # Check if the model supports predict_proba method for AUC calculation
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, proba[:, 1])
        else:  # Multiclass classification
            auc = roc_auc_score(y_test, proba, multi_class='ovo')
    else:
        auc = None

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
    
# Main function to execute the pipeline
def main():
    # Load trainWithLable data
    df = pd.read_csv('trainWithLabel.csv')
    
    # Preprocess the training data
    #preprocessor = Preprocessor(df)
    #df_processed = preprocessor.preprocess()
    #print(df_processed)

    # Define the models for classification
    models = {'Naive Bayes': NaiveBayesClassifier(),
              'KNN': KNearestNeighbors(),
              'MLP': MultilayerPerceptron()
    }

    # Split the dataset into features and target variable
    #X_train = df_processed.drop('Outcome', axis=1)
    #y_train = df_processed['Outcome']
    X_train = df.drop('Outcome', axis=1)
    y_train = df['Outcome']
    
    # Perform K-Fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = []

    for name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]#tain,val
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]#outcome
            preprocessor = Preprocessor(X_train_fold)
            X_train_fold_processed = preprocessor.preprocess()
            X_val_fold_processed = preprocessor.transform(X_val_fold)
            has_nan = X_val_fold_processed.isna().any().any()
            if has_nan:
                print("The dataset contains NaN values.")
            else:
                print("The dataset does not contain NaN values.")
            #print(X_val_fold_processed)
            model.fit(X_train_fold_processed, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold_processed, y_val_fold)
            fold_result['model'] = name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)

    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]

    print("Cross-validation results:")
    print(all_results_df)

    # Save results to an Excel file
    all_results_df.to_excel('cv_results.xlsx', index=False)
    print("Cross-validation results with averages saved to cv_results.xlsx")

    # Load the test dataset, assuming you have a test set CSV file without labels
    df_ = pd.read_csv('testWithoutLabel.csv')
    preprocessor_ = Preprocessor(df_)
    X_test = preprocessor_.preprocess()

    # Initialize an empty list to store the predictions of each model
    predictions = []

    # Make predictions with each model
    for name, model in models.items():
        model_predictions = model.predict(X_test)
        predictions.append({
            'model': name,
            'predictions': model_predictions
        })

    # Convert the list of predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predictions
    print("Model predictions:")
    print(predictions_df)

    # Save the predictions to an Excel file
    predictions_df.to_csv('test_results.csv', index=False)
    print("Model predictions saved to test_results.xlsx")

if __name__ == "__main__":
    main()
