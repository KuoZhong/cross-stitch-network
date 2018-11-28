if [ ! -d "office_data" ]; then
  mkdir office_data
else
  rm office_data/*
fi
cp  /opt/ml/disk/office/domain_adaptation_images.tar.gz office_data/
tar -zxvf office_data/domain_adaptation_images.tar.gz -C office_data
ls office_data
python train_progressive_net.py
rm -r office_data

