from setuptools import setup

setup(name='morph_disamb',
      version='1.0',
      description='Morphological disambiguator using recurrent or convolutional neural network',
      url='http://github.com/nagyniki017/morph_disamb',
      author='Nikolett Nagy',
      author_email='nikolettnagy017@gmail.com',
      license='MIT',
      packages=['morph_disamb'],
      install_requires=[
            'numpy >= 1.12.1',
            'tensorflow-gpu >= 1.1.0, <= 1.3.0',
            'Keras >= 2.0.4, <= 2.0.6',
            'h5py >= 2.7.0'
      ],
      include_package_data=True,
      package_data={
            'morph_disamb': ['save/*', 'data/*']
      })