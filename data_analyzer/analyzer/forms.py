from django import forms
from .models import DataSet

class DataSetForm(forms.ModelForm):
    class Meta:
        model = DataSet
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'file': forms.FileInput(attrs={'class': 'form-control', 'accept': '.xlsx,.xls'}),
        }

class AnalysisForm(forms.Form):
    ANALYSIS_CHOICES = [
        ('describe', 'آمار توصیفی'),
        ('correlation', 'همبستگی'),
        ('missing', 'داده‌های缺失'),
        ('outliers', 'داده‌های پرت'),
        ('distribution', 'توزیع داده‌ها'),
    ]
    
    analysis_type = forms.ChoiceField(
        choices=ANALYSIS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    columns = forms.MultipleChoiceField(
        required=False,
        widget=forms.SelectMultiple(attrs={'class': 'form-control'})
    )