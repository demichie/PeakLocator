from model import InputForm
from flask import Flask, render_template, request , make_response
from flask import send_from_directory
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.datastructures import CombinedMultiDict
import math
from pylab import *
from compute import compute
from werkzeug import secure_filename
import glob,os
import numpy as np


app = Flask(__name__)

# Relative path of directory for uploaded files
UPLOAD_DIR = 'uploads/'
DOWNLOAD_DIR = 'downloads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_DIR
app.secret_key = 'MySecretKey'

for file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'],"*.grd")):

    os.remove(file)


if os.path.isfile('./downloads/out.grd'):
    output_full = './downloads/out.grd'
    os.remove(output_full)


#WTF_CSRF_ENABLED = False

WTF_CSRF_SECRET_KEY = 'MySecretKey'

if not os.path.isdir(UPLOAD_DIR):
    os.mkdir(UPLOAD_DIR)


def save_grd(data):

    response = make_response(str(data))
    response.headers["Content-Disposition"] = "attachment; filename=result.grd"
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm(CombinedMultiDict((request.files, request.form)))
    filename = None  # default


    if request.method == 'POST':

        print 'values',request.form['submit']


        if ( request.form['submit'] == 'Compute' ) and form.validate_on_submit():
            print 'values',request.form['submit']
            print 'validate',form.validate_on_submit()

            for file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'],"*.grd")):

                os.remove(file)


            output_full = './downloads/out.grd'
            if os.path.isfile('./downloads/out.grd'):
                os.remove(output_full)

            filenames=[]
            MinMax=[]
            sigma=[]

            for field in form:
                # Make local variable (name field.name)
                if ( not('filename' in field.name) and

                    field.name != 'csrf_token'):
                    # exec('%s = ''%s''' % (field.name, field.data))
                    if ( 'MinMax' in field.name ):
                        MinMax.append(field.data)
                    if ( 'sigma' in field.name ):
                        sigma.append(field.data)


                elif ( 'filename' in field.name):
                    file = request.files[field.name]
                    if file:
                        # Make a valid version of filename for any file ystem
                        print 'load', file.filename
#                        filenames.append( secure_filename(file.filename) )
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                       secure_filename(file.filename)))

                        filenames.append(os.path.join(app.config['UPLOAD_FOLDER'],
                                       secure_filename(file.filename)))
            result = compute(filenames,MinMax,sigma)

            data_masked = result[3]
            header = result[4]

            data_masked[isnan(data_masked)] = 1.70141E+038

            np.savetxt(output_full, data_masked, header=header, newline='\n', fmt='%1.7f',comments='')

        elif ( request.form['submit'] == 'Save' ) and os.path.isfile('./downloads/out.grd'):

            print 'exist output',os.path.isfile('./downloads/out.grd')

            print 'values',request.form['submit']
            result = None
            print 'before save'
            return send_from_directory(directory='/home/demichie/downloads',filename='out.grd')


        else:

            result = None

    else:

        print 'validate',form.validate_on_submit()

        result = None

    return render_template('view.html', form=form,
                           result=result)

if __name__ == '__main__':
    app.run(debug=True,host= '0.0.0.0')
