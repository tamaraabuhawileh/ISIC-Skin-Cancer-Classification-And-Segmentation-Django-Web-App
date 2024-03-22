from django.db import models

# Create your models here.
class Case(models.Model):
    #image = models.FileField(upload_to='photos/%y/%m/%d')
    image = models.FileField(upload_to='photos')
    def __str__(self):
        return self.username
    
    class Meta:
        verbose_name = 'Past Cases'
        