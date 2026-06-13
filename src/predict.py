from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt

# Run predictions on validation set, then:
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])) # type: ignore

cm = confusion_matrix(y_true, y_pred) # type: ignore
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
plt.title('Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')