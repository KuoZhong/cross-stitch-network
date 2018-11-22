if [ ! -d "office_data" ]; then
  mkdir office_data
else
  rm office_data/*
fi
seven disk -download -src office/domain_adaptation_images.tar.gz -dest office_data/
tar -zxvf office_data/domain_adaptation_images.tar.gz -C office_data
ls office_data
python train_single_task.py
rm -r office_data

