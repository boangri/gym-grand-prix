import setuptools

setuptools.setup(name='gym-grand-prix',
                 version='0.2.2',
                 description='Gym GrandPrix Env',
                 url='https://github.com/boangri/gym-grand-prix',
                 author='Boris Gribovskiy',
                 packages=setuptools.find_packages(),
                 author_email='xinu@yandex.ru',
                 license='MIT License',
                 install_requires=['gym', 'numpy', 'pygame'],
                 python_requires='>=3.6'
                 )
