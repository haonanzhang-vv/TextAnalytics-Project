#export DATABASE_NAME=msia423_db

# docker run -it \
# --env MYSQL_HOST \
# --env MYSQL_PORT \
# --env MYSQL_USER \
# --env MYSQL_PASSWORD \
# --env DATABASE_NAME penny_mysql penny_lane_db.py 

# -p ${MYSQL_PORT}:${MYSQL_PORT} \ 
# echo "sup"
# echo  ${MYSQL_HOST}

docker run -it \
--env MYSQL_HOST \
--env MYSQL_PORT \
--env MYSQL_USER \
--env MYSQL_PASSWORD \
--env DATABASE_NAME \
tweet_sentiment_mysql src/db_models.py --RDS



