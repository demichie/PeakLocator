from wtforms import FloatField, validators
#from flask_wtf.file import FileField , FileRequired

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired , FileAllowed
from wtforms import SelectField

from math import pi
import functools


# Allowed file types for file upload
ALLOWED_EXTENSIONS = set(['grd', 'dat', 'npy'])

class InputForm(FlaskForm):

    myChoices = [('Max','max'),('Min','min')]

    sigmaChoices = [('1','1'),('2','2'),('3','3')]


    filename1 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS),FileRequired()])

    MinMax1 = SelectField('Min/Max', choices = myChoices)

    sigma1 = SelectField('sigma', choices = sigmaChoices)

    filename2 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS),FileRequired()])

    MinMax2 = SelectField(u'Min/Max', choices = myChoices)

    sigma2 = SelectField('sigma', choices = sigmaChoices)

    filename3 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax3 = SelectField(u'Min/Max', choices = myChoices)

    sigma3 = SelectField('sigma', choices = sigmaChoices)

    filename4 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax4 = SelectField(u'Min/Max', choices = myChoices)

    sigma4 = SelectField('sigma', choices = sigmaChoices)

    filename5 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax5 = SelectField(u'Min/Max', choices = myChoices)

    sigma5 = SelectField('sigma', choices = sigmaChoices)

    filename6 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax6 = SelectField('Min/Max', choices = myChoices)

    sigma6 = SelectField('sigma', choices = sigmaChoices)

    filename7 = FileField(
               label='Name of input file',
              validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax7 = SelectField(u'Min/Max', choices = myChoices)

    sigma7 = SelectField('sigma', choices = sigmaChoices)

    filename8 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax8 = SelectField(u'Min/Max', choices = myChoices)

    sigma8 = SelectField('sigma', choices = sigmaChoices)

    filename9 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax9 = SelectField(u'Min/Max', choices = myChoices)

    sigma9 = SelectField('sigma', choices = sigmaChoices)

    filename10 = FileField(
               label='Name of input file',
               validators=[FileAllowed(ALLOWED_EXTENSIONS)])

    MinMax10 = SelectField(u'Min/Max', choices = myChoices)

    sigma10 = SelectField('sigma', choices = sigmaChoices)

