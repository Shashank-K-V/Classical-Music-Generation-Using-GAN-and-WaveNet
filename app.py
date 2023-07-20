from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import os
from music21 import *
import os
app = Flask(__name__)
def read_midi(file):
    print("Loading Music File:",file)
    
    notes=[]
    notes_to_parse = None
    
    #parsing a midi file
    midi = converter.parse(file)
  
    #grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    #Looping over all the instruments
    for part in s2.parts:
    
        #select elements of only piano
        if 'Piano' in str(part): 
        
            notes_to_parse = part.recurse() 
      
            #finding whether a particular element is note or a chord
            for element in notes_to_parse:
                
                #note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                
                #chord
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/wave', methods=['POST'])
def generate_music():
    path='./Midi/'
    #read all the filenames
    files=[i for i in os.listdir(path) if i.endswith(".mid")]
    #reading each midi file
    notes_array = np.array([read_midi(path+i) for i in files])
    notes_ = [element for note_ in notes_array for element in note_]
    #No. of unique notes
    unique_notes = list(set(notes_))
    print(len(unique_notes))
    #importing library
    from collections import Counter

    #computing frequency of each note
    freq = dict(Counter(notes_))

    frequent_notes = [note_ for note_, count in freq.items() if count>=50]
    print(frequent_notes)
    print(len(frequent_notes))
    new_music=[]

    for notes in notes_array:
        temp=[]
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)            
        new_music.append(temp)
    
    new_music = np.array(new_music)
    new_music=[]

    for notes in notes_array:
        temp=[]
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)            
        new_music.append(temp)
        
    new_music = np.array(new_music)
    no_of_timesteps = 32
    x = []
    y = []

    for note_ in new_music:
        for i in range(0, len(note_) - no_of_timesteps, 1):
            
            #preparing input and output sequences
            input_ = note_[i:i + no_of_timesteps]
            output = note_[i + no_of_timesteps]
            
            x.append(input_)
            y.append(output)
            
    x=np.array(x)
    y=np.array(y)
    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
    x_seq=[]
    for i in x:
        temp=[]
        for j in i:
            #assigning unique integer to every note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)
        
    x_seq = np.array(x_seq)
    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y)) 
    y_seq=np.array([y_note_to_int[i] for i in y])
    from sklearn.model_selection import train_test_split
    x_tr, x_val, y_tr, y_val = train_test_split(x_seq,y_seq,test_size=0.2,random_state=0)
    from keras.layers import (Dense,
                                Flatten,Conv1D,Embedding,MaxPool1D,Dropout,GlobalMaxPool1D)
    from keras.models import Sequential
    from keras.callbacks import ModelCheckpoint
    #import keras.backend as K

    #K.clear_session()
    model = Sequential()
        
    #embedding layer
    model.add(Embedding(len(unique_x), 100, input_length=32,trainable=True)) 

    model.add(Conv1D(64,3, padding='causal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
        
    model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
            
    #model.add(Conv1D(256,5,activation='relu'))    
    model.add(GlobalMaxPool1D())
        
    model.add(Dense(256, activation='relu'))
    model.add(Dense(len(unique_y), activation='softmax'))
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])

    model.summary()
    mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)
    from keras.callbacks import ModelCheckpoint
    filepath = "./Model/model-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath,monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'max')
    callbacks_list = [checkpoint]
    history = model.fit(np.array(x_tr),np.array(y_tr),
                        batch_size=128,epochs=5, verbose=1,validation_data=(np.array(x_val),np.array(y_val))
                        )
    import random
    ind = np.random.randint(0,len(x_val)-1)
    ind1 = np.random.randint(0,len(x_val)-1)

    random_music = x_val[ind]
    random_music1 = x_val[ind1]

    predictions=[]
    for i in range(10):

        random_music = random_music.reshape(1,no_of_timesteps)

        prob  = model.predict(random_music)[0]
        y_pred= np.argmax(prob,axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
        random_music = random_music[1:]
    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x)) 
    predicted_notes = [x_int_to_note[i] for i in random_music]
    def convert_to_midi(prediction_output):
    
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    
                    cn=int(current_note)
                    new_note = note.Note(cn)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                    
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
                
            # pattern is a note
            else:
                
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 1
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='static/music/wave.mid')
    convert_to_midi(predicted_notes)
    wmusic_url = url_for('static', filename='music/wave.mid')
    return render_template('wavemusic.html',wmusic_url=wmusic_url)

@app.route('/gan', methods=['POST'])
def gan_generate():
    import tensorflow as tf
    import numpy as np
    import glob
    import music21

    # Define the generator and discriminator models
    def make_generator_model(input_shape, output_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, input_shape=input_shape, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='tanh'))
        model.add(tf.keras.layers.Reshape(output_shape))
        return model

    def make_discriminator_model(input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    # Define the loss functions for the generator and discriminator
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    # Define the optimizers for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Define the training loop for the GAN
    @tf.function
    def train_step(real_music, generator, discriminator):
        # Generate some random noise
        noise = tf.random.normal([len(real_music), 100])

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            generated_music = generator(noise, training=True)
            real_output = discriminator(real_music, training=True)
            fake_output = discriminator(generated_music, training=True)
            disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            generated_music = generator(noise, training=True)
            fake_output = discriminator(generated_music, training=True)
            gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Load the MIDI files
    midi_files = glob.glob('Midi/*.mid')
    print("files")
    midi_data = []
    for file in midi_files:
        midi = converter.parse(file)
        midi_data.append(midi)

    print(midi_data)

    # Preprocess the MIDI data
    input_shape = (100,)
    output_shape = (128, 128)
    x_train = np.zeros((len(midi_data), *input_shape))
    y_train = np.zeros((len(midi_data), *output_shape))
    for i, midi in enumerate(midi_data):
        midi_arr = np.array([note.pitches[0].midi for note in midi.flat.notes])
        if len(midi_arr) > 0:
            x_train[i] = np.random.normal(size=input_shape)
            y_train[i, :len(midi_arr), midi_arr] = 1

    # Create the generator and discriminator
    generator = make_generator_model(input_shape=(100,), output_shape=(128, 128))
    discriminator = make_discriminator_model(input_shape=(128, 128))

    # Train the GAN for 100 epochs
    batch_size = 32
    num_epochs = 100
    num_batches = len(x_train) // batch_size

    for epoch in range(num_epochs):
        np.random.shuffle(x_train)
        for i in range(num_batches):
            real_data = x_train[i*batch_size:(i+1)*batch_size]
            train_step(real_data)
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{num_batches}, Disc Loss: {disc_loss}, Gen Loss: {gen_loss}')

    # Generate some new MIDI data using the trained generator
    noise = tf.random.normal([1, *input_shape])
    generated_data = generator(noise).numpy()
    print(generated_data)
    import music21 as m21
    import numpy as np

    def decode_output(output, output_shape):
        # Extract the number of notes and pitch classes
        num_notes, num_pitch_classes = output_shape
        
        # Initialize the MIDI data
        midi_data = m21.stream.Stream()
        
        # Iterate over the output and convert each chord back into a Music21 chord object
        for i in range(num_notes):
            # Extract the notes of the chord and convert them from one-hot to integer encoding
            chord_notes = np.argmax(output[0, i, :])
            
            # If the chord is not a rest, create a Music21 chord object and append it to the MIDI data
            if chord_notes != 0:
                note_chord = m21.chord.Chord()
                note_chord.add(m21.note.Note(chord_notes))
                midi_data.append(note_chord)
            else:
                midi_data.append(m21.note.Rest())
        
        # Return the MIDI data
        return midi_data

    decoded_output = decode_output(generated_data, output_shape)
    midi_stream = stream.Stream()
    for chord in decoded_output:
        midi_stream.append(chord)
    midi_stream.write('midi', fp='static/music/gan.mid')
    gmusic_url = url_for('static', filename='music/gan.mid')
    return render_template('ganmusic.html',gmusic_url=gmusic_url)

@app.route('/index', methods=['POST'])
def index_home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
