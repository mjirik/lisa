#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Processing of exceptions with lisa. """

import logging
logger = logging.getLogger(__name__)


import traceback
# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText


def reportException(exception):
    excstr = traceback.format_exc()
    try:
        sendMail(excstr, 'Lisa exception: ' + str(exception))
    except Exception as e:
        logger.debug("Problems with sending exception report")
        logger.debug(traceback.format_exc())
        # logger.debug(str(e))
        # logger.debug("Original exception:")
        # logger.debug(str(exception))
        # logger.debug(excstr)

    logger.exception(excstr)

    raise(exception)


def sendMail(mailcontent, subject='None'):
    me = 'mjirik@kky.zcu.cz'
    you = 'miroslav.jirik@gmail.com'

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# fp = open(textfile, 'rb')
# # Create a text/plain message
    msg = MIMEText(mailcontent)

# me == the sender's email address
# you == the recipient's email address
    msg['Subject'] = subject
    msg['From'] = me
    msg['To'] = you

# Send the message via our own SMTP server, but don't include the
# envelope header.
    s = smtplib.SMTP('localhost')
    s.sendmail(me, [you], msg.as_string())
    s.quit()
    logger.debug('Subject: ', subject)
    logger.warn('Mail content')
    logger.debug(mailcontent)
