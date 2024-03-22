from .models import Case
from django import forms

class CaseForm(forms.ModelForm):
    class Meta:
        model = Case
        fields = {'image'}