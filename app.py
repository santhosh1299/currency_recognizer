
  
import flask
import werkzeug
import requests


# Keras
from tensorflow import keras
import numpy as np
import tensorflow
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img

# Flask utils




app = flask.Flask(__name__)
API_KEY = 'EKHHX2LBC5QL76OX'

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    filename = "android.jpg"
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    """
    denom ="0"
    code ="USD"
    coun ="-"
    
    model_predict(img_path, model,model_2)
    hio =convertor(denom,code)
    own_value = str(hio)
    country_code = "USD"
    deno_code = "100"
    out  = "\n\nCountry : "+coun+"\n\nDeno : "+denom+"\n\nNative : "+own_value 

    """

    out = model_predict(img_path, model,model_2)
    final_deno=str(out[0])
    final_country =str(out[1])
    print(final_country)
    final_own = str(convertor(out[0],out[2]))
    if final_country==" India ":
        Final_string =  "\nCountry : "+final_country+"\n\nDeno : "+final_deno  
    else:  
        Final_string =  "\nCountry : "+final_country+"\n\nDeno : "+final_deno+"\n\nNative : "+final_own+".Rs"
    print(final_country,final_deno)
    return (Final_string)


def convertor(amount,code):
    amount = float(amount)
    
    from_c =code
    
    to_c ="INR"
    url = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={}&to_currency={}&apikey={}'.format( 
            from_c, to_c, API_KEY)
    response = requests.get(url=url).json()
    rate = response['Realtime Currency Exchange Rate']['5. Exchange Rate'] 
    rate = float(rate) 
    result = rate * amount 
   
    return round(result)

# Model saved with Keras model.save()
# 7countries
curr_PATH ='D:/flask_app/models/currencymodel_project_1.h5'
country_PATH2 = 'D:/flask_app/models/country_updated_proj.h5' 

# lastnight 1
'''curr_PATH ='D:/flask_app/models/epochs_5_dropout.h5'
country_PATH2 = 'D:/flask_app/models/model_up_1_country.h5' '''
'''curr_PATH ='D:/flask_app/models/3_country_deno.h5'
country_PATH2 = 'D:/flask_app/models/3_country.h5' '''
img_path = 'D:/flask_app/android.jpg'
# Load your trained model
model = load_model(curr_PATH)
model_2 = load_model(country_PATH2)   


def model_predict(img_path, model,model_2):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = np.expand_dims(x, axis=0)
   
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    deno = model.predict(x)
    deno=deno.argmax()

    country = model_2.predict(x)
    country = country.argmax()
    #last night 1
    ''' curr_dict = {0:"1",1:"10",2:"100", 3:"1000",4:"2",5:"20", 6:"200",7:"2000",8:"5",9:"50",10:"500"}
    coun_dict = {0:" Europe ", 1:" India ", 2:"Singapore",3:"United Kingdom"}
    code_dict = {0:"EUR", 1:"INR", 2:"SGD",3:"GBP"}  '''



    curr_dict = {0:"1",1:"10",2:"100", 3:"1000",4:"2",5:"20", 6:"200",7:"2000",8:"5",9:"50",10:"500"}
    coun_dict = {0:'Europe', 1:'India', 2:'Singapore', 3: 'UK pound'}
    code_dict = {0:"EUR", 1:"INR", 2:"SGD",3:"GBP"}


    #7 countries
    '''curr_dict = {0:"1",1:"1-2",2:"1-4", 3:"10",4:"100",5:"1000", 6:"10000",7:"20",8:"200",9:"5",10:"50",11:"500",12:"1000"}
    code_dict = {0:"AUD", 1:"EUR", 2:"JPY",3:"KWD",4:"MXN",5:"SGD",6:"CHF",7:"GBP"}
    coun_dict = {0:"Australia", 1:"Europe", 2:"Japan",3:"Kuwait",4:"Mexico",5:"Singapore",6:"Switzerland",7:"United Kingdom"} '''

    '''curr_dict = {0:"1",1:"10",2:"100", 3:"1000",4:"2",5:"20", 6:"200",7:"2000",8:"5",9:"50",10:"500"}
     
    #code_dict = {0:"AUD", 1:"EUR", 2:"INR",3:"JPY",4:"KWD",5:"MXN",6:"NZD",7:"SGD",8:"CHF",9:"GBP",10:"USD"}
    #coun_dict = {0:"Australia", 1:"Europe", 2:"India",3:"Japan",4:"Kuwait",5:"Mexico",6:"New Zealand",7:"Singapore",8:"Switzerland",9:"United Kingdom",10:"United States Of America"}'''



    denom =curr_dict.get(deno)
    code = code_dict.get(country)
    coun = coun_dict.get(country)
    
    
    return denom,coun,code
    
    


app.run(host="0.0.0.0", port=5000, debug=True)