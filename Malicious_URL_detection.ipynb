{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.grid_search import GridSearchCV\n",
    "from sklearn.cross_validation import cross_val_score\n",
    "from sklearn import metrics\n",
    "from sklearn.model_selection import learning_curve\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>url</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>diaryofagameaddict.com</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>espdesign.com.au</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>iamagameaddict.com</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>kalantzis.net</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>slightlyoffcenter.net</td>\n",
       "      <td>bad</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      url label\n",
       "0  diaryofagameaddict.com   bad\n",
       "1        espdesign.com.au   bad\n",
       "2      iamagameaddict.com   bad\n",
       "3           kalantzis.net   bad\n",
       "4   slightlyoffcenter.net   bad"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = pd.read_csv('./data/data.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'q': ['abc'], 'p': ['123']}\n"
     ]
    }
   ],
   "source": [
    "import urlparse\n",
    "url = 'http://example.com/?q=abc&p=123'\n",
    "par = urlparse.parse_qs(urlparse.urlparse(url).query)\n",
    "\n",
    "print par\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def getTokens(input):\n",
    "    string = str(input.encode('utf-8'))\n",
    "    tokenList = filter(None, re.split(r\"[\\./]\", string))\n",
    "\n",
    "    tokenList = list(set(tokenList))\n",
    "    if 'com' in tokenList:\n",
    "        tokenList.remove('com')\n",
    "    if 'net' in tokenList:\n",
    "        tokenList.remove('net')\n",
    "    return tokenList\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "corpus = [d for d in data['url']]\n",
    "y = [[1,0][d!= 'bad'] for d in data['label']]\n",
    "vectorizer2 = TfidfVectorizer(tokenizer=getTokens)\n",
    "X = vectorizer2.fit_transform(corpus)\n",
    "\n",
    "idf2 = vectorizer2.idf_\n",
    "#print dict(zip(vectorizer2.get_feature_names(), idf2))\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=None, error_score='raise',\n",
       "       estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n",
       "          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,\n",
       "          verbose=0, warm_start=False),\n",
       "       fit_params={}, iid=True, n_jobs=-1,\n",
       "       param_grid={'penalty': ['l1', 'l2'], 'C': array([   0.1,    1. ,   10. ,  100. ])},\n",
       "       pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lgs = LogisticRegression()\n",
    "param = {'C': np.logspace(-1, 2, 4),'penalty':['l1','l2']}\n",
    "clf = GridSearchCV(lgs, param, n_jobs = -1)\n",
    "clf.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "best score: 0.972087\n",
      "best parameters: {'penalty': 'l2', 'C': 100.0}\n",
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       1.00      1.00      1.00    275692\n",
      "          1       1.00      1.00      1.00     60679\n",
      "\n",
      "avg / total       1.00      1.00      1.00    336371\n",
      "\n",
      "[[275685      7]\n",
      " [     2  60677]]\n",
      "0.974073268331\n",
      "best score: 0.972087\n",
      "best parameters: {'penalty': 'l2', 'C': 100.0}\n",
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       0.98      1.00      0.99     69129\n",
      "          1       0.98      0.88      0.93     14964\n",
      "\n",
      "avg / total       0.98      0.98      0.98     84093\n",
      "\n",
      "[[68887   242]\n",
      " [ 1727 13237]]\n",
      "0.958522147119\n"
     ]
    }
   ],
   "source": [
    "def getAccuracy(clf, train, target):\n",
    "    print \"best score: %f\"%clf.best_score_\n",
    "    print \"best parameters: %s\"%clf.best_params_\n",
    "    predict = clf.best_estimator_.predict(train)\n",
    "    print(metrics.classification_report(target,predict))\n",
    "    print(metrics.confusion_matrix(target, predict))\n",
    "    print(cross_val_score(clf.best_estimator_, train, target, cv= 5).mean())\n",
    "getAccuracy(clf, X_train, y_train)\n",
    "getAccuracy(clf, X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.976585447065\n",
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0       0.98      1.00      0.99     69129\n",
      "          1       0.98      0.88      0.93     14964\n",
      "\n",
      "avg / total       0.98      0.98      0.98     84093\n",
      "\n",
      "[[68887   242]\n",
      " [ 1727 13237]]\n",
      "0.956215142759\n"
     ]
    }
   ],
   "source": [
    "lgs2 = LogisticRegression(C = 100, penalty= 'l2')\n",
    "lgs2.fit(X_train, y_train)\n",
    "print(lgs2.score(X_test, y_test))\n",
    "predict2 = lgs2.predict(X_test)\n",
    "\n",
    "print metrics.classification_report(y_test, predict2)\n",
    "print(metrics.confusion_matrix(y_test, predict2))\n",
    "print cross_val_score(lgs2, X_test, y_test).mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 1 1 1 1]\n"
     ]
    }
   ],
   "source": [
    "X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']\n",
    "X_predict = vectorizer2.transform(X_predict)\n",
    "y_Predict = lgs2.predict(X_predict)\n",
    "print y_Predict #printing predicted values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\n",
    "def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,\n",
    "                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):\n",
    "    plt.figure(figsize=(10,6))\n",
    "    plt.title(title)\n",
    "    if ylim is not None:\n",
    "        plt.ylim(*ylim)\n",
    "    plt.xlabel(\"Training examples\")\n",
    "    plt.ylabel(scoring)\n",
    "    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,\n",
    "                                                            n_jobs=n_jobs, train_sizes=train_sizes)\n",
    "    train_scores_mean = np.mean(train_scores, axis=1)\n",
    "    train_scores_std = np.std(train_scores, axis=1)\n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    test_scores_std = np.std(test_scores, axis=1)\n",
    "    plt.grid()\n",
    "\n",
    "    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\n",
    "                     train_scores_mean + train_scores_std, alpha=0.1,\n",
    "                     color=\"r\")\n",
    "    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,\n",
    "                     test_scores_mean + test_scores_std, alpha=0.1, color=\"g\")\n",
    "    plt.plot(train_sizes, train_scores_mean, 'o-', color=\"r\",\n",
    "             label=\"Training score\")\n",
    "    plt.plot(train_sizes, test_scores_mean, 'o-', color=\"g\",\n",
    "             label=\"Cross-validation score\")\n",
    "\n",
    "    plt.legend(loc=\"best\")\n",
    "    return plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x118c247d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_learning_curve(lgs2, 'Logistic Regression', X_train, y_train, cv=4);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_curve, auc\n",
    "def plot_roc_curve(estimator, X, y, title):\n",
    "    # Determine the false positive and true positive rates\n",
    "    fpr, tpr, _ = roc_curve(y, estimator.predict_proba(X)[:,1])\n",
    "\n",
    "    # Calculate the AUC\n",
    "    roc_auc = auc(fpr, tpr)\n",
    "    print ('ROC AUC: %0.2f' % roc_auc)\n",
    "\n",
    "    # Plot of a ROC curve for a specific class\n",
    "    plt.figure(figsize=(10,6))\n",
    "    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)\n",
    "    plt.plot([0, 1], [0, 1], 'k--')\n",
    "    plt.xlim([0.0, 1.0])\n",
    "    plt.ylim([0.0, 1.05])\n",
    "    plt.xlabel('False Positive Rate')\n",
    "    plt.ylabel('True Positive Rate')\n",
    "    plt.title('ROC Curve - {}'.format(title))\n",
    "    plt.legend(loc=\"lower right\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC: 0.99\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11b672e90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_roc_curve(lgs2, X_test, y_test, 'Logistic Regression')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "420464"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,\n",
    "                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):\n",
    "    plt.figure(figsize=(10,6))\n",
    "    plt.title(title)\n",
    "    if ylim is not None:\n",
    "        plt.ylim(*ylim)\n",
    "    plt.xlabel(\"Training examples\")\n",
    "    plt.ylabel(scoring)\n",
    "    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,\n",
    "                                                            n_jobs=n_jobs, train_sizes=train_sizes)\n",
    "    train_scores_mean = np.mean(train_scores, axis=1)\n",
    "    train_scores_std = np.std(train_scores, axis=1)\n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    test_scores_std = np.std(test_scores, axis=1)\n",
    "    plt.grid()\n",
    "\n",
    "    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\n",
    "                     train_scores_mean + train_scores_std, alpha=0.1,\n",
    "                     color=\"r\")\n",
    "    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,\n",
    "                     test_scores_mean + test_scores_std, alpha=0.1, color=\"g\")\n",
    "    plt.plot(train_sizes, train_scores_mean, 'o-', color=\"r\",\n",
    "             label=\"Training score\")\n",
    "    plt.plot(train_sizes, test_scores_mean, 'o-', color=\"g\",\n",
    "             label=\"Cross-validation score\")\n",
    "\n",
    "    plt.legend(loc=\"best\")\n",
    "    return plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "\n",
    "# and plot the result\n",
    "plt.figure(1, figsize=(4, 3))\n",
    "plt.clf()\n",
    "plt.scatter(idf2, y, color='black', zorder=20)\n",
    "X_test = np.linspace(-5, 10, 300)\n",
    "\n",
    "\n",
    "def model(x):\n",
    "    return 1 / (1 + np.exp(-x))\n",
    "loss = model(X_test * clf.coef_ + clf.intercept_).ravel()\n",
    "plt.plot(X_test, loss, color='red', linewidth=3)\n",
    "\n",
    "ols = linear_model.LinearRegression()\n",
    "ols.fit(X, y)\n",
    "plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)\n",
    "plt.axhline(.5, color='.5')\n",
    "\n",
    "plt.ylabel('y')\n",
    "plt.xlabel('X')\n",
    "plt.xticks(range(-5, 10))\n",
    "plt.yticks([0, 0.5, 1])\n",
    "plt.ylim(-.25, 1.25)\n",
    "plt.xlim(-4, 10)\n",
    "plt.legend(('Logistic Regression Model', 'Linear Regression Model'),\n",
    "           loc=\"lower right\", fontsize='small')\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
