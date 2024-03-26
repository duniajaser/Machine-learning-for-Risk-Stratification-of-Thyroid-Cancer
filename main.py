import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


def main():
    file_path = r'C:\Users\saad\Desktop\Machine project\Thyroid_Diff.csv'
    thyroid_data = pd.read_csv(file_path)
    print("--------------------------------------------------")
    data_summary = {
        'Number of Rows': thyroid_data.shape[0],
        'Number of Columns': thyroid_data.shape[1],
        'Columns': thyroid_data.columns.tolist(),
        'Missing Values': thyroid_data.isnull().sum().to_dict(),
        'Data Types': thyroid_data.dtypes.to_dict()
    }
    for key, value in data_summary.items():
        print(f'{key}: {value}')
    print("--------------------------------------------------")
    # statistics
    # Descriptive statistics for numerical columns
    numerical_stats = thyroid_data.describe()
    # Frequency distribution for categorical columns
    categorical_columns = thyroid_data.select_dtypes(include=['object']).columns
    categorical_stats = thyroid_data[categorical_columns].describe()
    print("Descriptive Statistics for Numerical Columns:")
    print(numerical_stats)
    print("--------------------------------------------------")
    print("Descriptive Statistics for Categorical Columns:")
    print(categorical_stats)
    print("--------------------------------------------------")
    # visualization
    X_train, X_test, y_train, y_test = preprocess_data(thyroid_data)
    bar_box_plot(thyroid_data)
    combined_age_histogram(thyroid_data)
    box_plot(thyroid_data)
    # Baseline Model section
    baseline_model(thyroid_data,X_train, X_test, y_train, y_test)
    # we will use SVM and Decision Tree
    svm_modeling(X_train, X_test, y_train, y_test)
    decision_tree_accuracies(X_train, y_train, X_test, y_test)


# This function performs baseline modeling on thyroid data.
def baseline_model(thyroid_data, X_train, X_test, y_train, y_test):
    # Evaluating KNN for k=1 and k=3
    k_values = [1, 3, 5, 7, 9, 11, 12, 15, 18, 20, 25]
    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        accuracy_train, accuracy_test = evaluate_knn(k, X_train, X_test, y_train, y_test)
        train_accuracies.append(accuracy_train)
        test_accuracies.append(accuracy_test)

    # Plotting the performance as lines
    plt.figure(figsize=(8, 4))
    plt.plot(k_values, train_accuracies, marker='o', label='Train Accuracy', color='blue')
    plt.plot(k_values, test_accuracies, marker='o', label='Test Accuracy', color='orange')

    # Annotating each point with its corresponding accuracy value
    for i in range(len(k_values)):
        plt.text(k_values[i], train_accuracies[i], f"{train_accuracies[i]:.2f}", fontsize=8)
        plt.text(k_values[i], test_accuracies[i], f"{test_accuracies[i]:.2f}", fontsize=8)

    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('KNN Performance for Different K Values')
    plt.legend(loc='upper right')

    plt.show()

    return train_accuracies, test_accuracies


# Function to evaluate KNN with different values of k
def evaluate_knn(k, X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    return accuracy_train, accuracy_test


def bar_box_plot(thyroid_data):
    code_to_pathology = {2: 'Micropapillary', 3: 'Papillary', 0: 'Follicular', 1: 'Hurthel cell'}
    # Replace the integer codes with the corresponding string values
    thyroid_data['Pathology'] = thyroid_data['Pathology'].replace(code_to_pathology)
    # Ensure 'Pathology' column is the index of the DataFrame
    thyroid_data.set_index('Pathology', inplace=True)
    # Modify the 'Risk' column values in the DataFrame
    thyroid_data['Risk'] = thyroid_data['Risk'].replace({0: 'High', 1: 'Intermediate', 2: 'Low'})
    # Create a custom color palette for 'Risk' categories
    risk_palette = {'High': '#4c99cf', 'Intermediate': '#65cf7c', 'Low': '#cf65a4'}
    # Create the pivot table
    pivot_table = thyroid_data.pivot_table(index='Pathology', columns='Risk', aggfunc='size', fill_value=0)
    # Normalize the pivot_table by the total cases to get the proportion
    normalized_pivot = pivot_table.div(pivot_table.sum(axis=1), axis=0)
    # Plotting the normalized stacked bar chart with custom color palette
    ax = normalized_pivot.plot(kind='bar', stacked=True, figsize=(10, 8),
                               color=[risk_palette[risk] for risk in normalized_pivot.columns], edgecolor='black')

    # Setting the labels and title
    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_ylabel('Proportion of Risk', fontsize=12)
    ax.set_title('Proportion of Risk Levels for Different Pathologies', fontsize=14)
    # Set the y-axis limits
    ax.set_ylim(0, 1)
    # Set the x-tick labels to the pathology names, ensuring they are properly displayed
    ax.set_xticklabels(normalized_pivot.index, rotation=45, ha="right")
    # Show the plot
    plt.tight_layout()
    plt.show()


def combined_age_histogram(thyroid_data):
    # Create separate data arrays for each gender
    thyroid_data['Gender'] = thyroid_data['Gender'].replace({0: 'F', 1: 'M'})
    male_age = thyroid_data[thyroid_data['Gender'] == 'M']['Age']
    thyroid_data['Risk'] = thyroid_data['Risk'].replace({0: 'High', 1: 'Intermediate', 2: 'Low'})
    female_age = thyroid_data[thyroid_data['Gender'] == 'F']['Age']
    # Create a histogram for male ages
    plt.hist(male_age, bins=10, alpha=0.5, label='Male', color='blue', edgecolor='black')
    # Create a histogram for female ages
    plt.hist(female_age, bins=10, alpha=0.5, label='Female', color='pink', edgecolor='black')
    # Add labels, legend, and title
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Combined Age Distribution by Gender')

    # Show the plot
    plt.show()


def box_plot(thyroid_data):
    thyroid_data['Stage'] = thyroid_data['Stage'].replace({0: 'I', 1: 'II', 2: 'III', 3: 'IVA', 4: 'IVB'})

    # Define a color palette that generates distinct colors for each stage
    stage_palette = sns.color_palette('Set2', len(thyroid_data['Stage'].unique()))

    # Create a box plot using Seaborn with the specified palette
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Stage', y='Age', data=thyroid_data, hue='Stage', palette=stage_palette, legend=False)

    plt.title('Box Plot of Age by Stage')
    plt.xlabel('Stage')
    plt.ylabel('Age')

    # Remove the legend
    plt.legend([], [], frameon=False)

    plt.show()


def svm_modeling(X_train, X_test, y_train, y_test):
    C_values = [0.1, 1, 10, 100]
    # For storing metrics
    metrics = {
        'C': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'Cross-Validation Score': []
    }

    for C in C_values:
        svm_model = SVC(kernel='linear', C=C)

        # Perform cross-validation on the training data
        cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
        metrics['Cross-Validation Score'].append(cv_scores.mean())

        # Training the model
        svm_model.fit(X_train, y_train)

        # Making predictions
        y_train_pred = svm_model.predict(X_train)
        y_test_pred = svm_model.predict(X_test)

        # Calculating metrics
        metrics['C'].append(C)
        metrics['Train Accuracy'].append(accuracy_score(y_train, y_train_pred))
        metrics['Test Accuracy'].append(accuracy_score(y_test, y_test_pred))
        metrics['Precision'].append(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
        metrics['Recall'].append(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
        metrics['F1 Score'].append(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))

        # Printing classification report for each C
        print(f"\nClassification report for C={C}:")
        print(classification_report(y_test, y_test_pred, zero_division=0))
    print("--------------------------------------------------")
    print("SVM Accuracies:", metrics['Train Accuracy'])
    print("--------------------------------------------------")

    # Plotting accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['C'], metrics['Train Accuracy'], label='Train Accuracy', marker='o')
    plt.plot(metrics['C'], metrics['Test Accuracy'], label='Test Accuracy', marker='o')
    plt.title('SVM Accuracies for Different C Values')
    plt.xlabel('C Value')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.show()


def preprocess_data(thyroid_data):
    # Preprocessing the dataset
    # Encoding categorical variables
    label_encoders = {}
    for column in thyroid_data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        thyroid_data[column] = le.fit_transform(thyroid_data[column])
        label_encoders[column] = le
    # Choosing 'Recurred' as the target variable for prediction
    X = thyroid_data.drop('Recurred', axis=1)
    y = thyroid_data['Recurred']
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def decision_tree_accuracies(X_train, y_train, X_test, y_test):
    max_depth_values = [2, 4, 6, 8, 10]
    tree_train_accuracies = []
    tree_test_accuracies = []
    best_max_depth = None
    best_test_accuracy = 0.0
    best_tree_model = None

    for max_depth in max_depth_values:
        tree_model = DecisionTreeClassifier(max_depth=max_depth)
        tree_model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, tree_model.predict(X_train))
        test_accuracy = accuracy_score(y_test, tree_model.predict(X_test))

        tree_train_accuracies.append(train_accuracy)
        tree_test_accuracies.append(test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_max_depth = max_depth
            best_tree_model = tree_model
    # print("--------------------------------------------------")
    # print(f"The best test accuracy is {best_test_accuracy:.2f} at max depth {best_max_depth}.")
    # print("--------------------------------------------------")

    # Function to map feature indices to names
    def feature_index_to_name(index):
        if index < 0 or index >= len(X_train.columns):
            return "Invalid feature index"
        return X_train.columns[index]

    # Print nodes of the best tree model
    def print_tree_nodes(node, depth=0):
        indent = "  " * depth
        if best_tree_model.tree_.children_left[node] == best_tree_model.tree_.children_right[node]:
            print(
                f"{indent}Leaf {node}: class={best_tree_model.classes_[best_tree_model.tree_.value[node].argmax()]} count={best_tree_model.tree_.n_node_samples[node]}")
        else:
            feature_name = feature_index_to_name(best_tree_model.tree_.feature[node])
            print(
                f"{indent}Node {node}: feature={feature_name} (index {best_tree_model.tree_.feature[node]}), threshold={best_tree_model.tree_.threshold[node]}, count={best_tree_model.tree_.n_node_samples[node]}")
            print_tree_nodes(best_tree_model.tree_.children_left[node], depth + 1)
            print_tree_nodes(best_tree_model.tree_.children_right[node], depth + 1)

    # Print nodes of the best tree model
    print(f"Best Decision Tree (Max Depth {best_max_depth}) Nodes:")
    print_tree_nodes(0)

    # Plot accuracies
    plt.plot(max_depth_values, tree_train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(max_depth_values, tree_test_accuracies, label='Test Accuracy', marker='o')
    plt.title('Decision Tree Accuracies')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot the decision tree of the best model
    plot_tree(best_tree_model, filled=True, feature_names=X_train.columns,
              class_names=[str(c) for c in best_tree_model.classes_])
    plt.title(f'Best Decision Tree (Max Depth {best_max_depth})')
    plt.show()

    # Plotting accuracies
    plt.plot(max_depth_values, tree_train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(max_depth_values, tree_test_accuracies, label='Test Accuracy', marker='o')
    plt.title('Decision Tree Accuracies for Different Depth Values')
    plt.xlabel('Depth Values')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
