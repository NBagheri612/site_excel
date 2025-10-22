from django.db import models
import os

def upload_to(instance, filename):
    return f'datasets/{filename}'

class DataSet(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to=upload_to)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns = models.JSONField(default=list, blank=True)
    row_count = models.IntegerField(default=0)
    
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

class AnalysisResult(models.Model):
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=100)
    result_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.dataset.name} - {self.analysis_type}"