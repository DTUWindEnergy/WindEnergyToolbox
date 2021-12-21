#!/bin/bash

find ./ -type f -iname "*.py" -exec sed -i '/from __future__ import print_function/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from __future__ import division/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from __future__ import unicode_literals/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from __future__ import absolute_import/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from io import open/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import object/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import map/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import chr/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import dict/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import super/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import zip/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import range/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import str/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from builtins import int/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/from future import standard_library/d' {} \;
find ./ -type f -iname "*.py" -exec sed -i '/standard_library.install_aliases()/d' {} \;

