# Generated by Django 4.2.5 on 2023-12-16 11:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pages', '0011_alter_case_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='case',
            name='password',
        ),
        migrations.RemoveField(
            model_name='case',
            name='username',
        ),
    ]
