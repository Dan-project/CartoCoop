# CartoCoop - Cartographie sémantique coopérative multirobots

Ce dépôt regroupe le travail réalisé au laboratoire CRIStAL, dans le cadre du projet ANR-SOS, sur la cartographie sémantique multirobots par fusion LiDAR-caméra.

## Contenu du dépôt

| Dossier | Description |
|---|---|
| `losy/` | Package ROS2 LOSY (LIO-SAM Semantic YOLO), entièrement développé dans ce projet. Il construit une carte sémantique à partir d'un LiDAR, d'une caméra et de l'odométrie de LIORF. |
| `liorf/` | Version de [LIORF](https://github.com/YJZLuckyBoy/liorf), lui-même dérivé de LIO-SAM. Quelques fichiers de configuration et scripts ont été ajoutés pour l'adapter à nos robots (Scout et Ranger). |
| `Prior-Map/` | Outil de création de la carte a priori : à partir d'un export GeoJSON d'OpenStreetMap, il produit une carte 2D discrétisée et labellisée, au format GeoJSON. |

Les deux packages ROS2, LOSY et LIORF, fonctionnent ensemble : LIORF fournit l'odométrie et la détection de loop closure, LOSY y ajoute l'information sémantique.

## Documentation

- `losy/README.md` : présentation de LOSY et commandes de lancement
- `losy/README_DEV.md` : description des fonctions, pour les développeurs
- `Prior-Map/README.md` : création de la carte a priori

# Connexion docker sur le pc du labo 

Dans multirobots_sim/ :
- lancer le conteneur (vérifier le nom du conteneur (docker ps -a), normalement carto):

  `docker start carto`

  `docker exec -it carto bash`
  
Plus d'informations sur le simulateur ici: https://github.com/ccappelle-Ulille/SOS_multirobots_simulator.

