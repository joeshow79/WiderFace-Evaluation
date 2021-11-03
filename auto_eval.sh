# Detect on WIDER val dataset and output the result txt in place
rm -rf WIDER_t/
cp -r WIDER_val/ WIDER_t
python2 test_ocv_face_on_wider.py

# Evaluate the output compare to GT
rm -rf ocv_output
cp -r WIDER_t ocv_output
find ocv_output -iname "*.jpg" | xargs rm -rf
python evaluation.py

