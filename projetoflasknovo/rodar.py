"""Script auxiliar para gerar um token hexadecimal aleatório de 256 bits."""

import os


print(os.urandom(32).hex())
